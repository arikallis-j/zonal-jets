import numpy as np

from .const import *

class Atmosphere:
    def __init__(self, parameters = None, grid = None, method = None, initial = None, forcing = None, alpha = 0):
        self._initials = {
            "zero": self._s_zero,
            "monopole": self._s_monople,
            "dipole": self._s_dipole,
            "random": self._s_random,
        }

        self._forcings = {
            "zero": self._xi_zero,
            "random": self._xi_random,
        }

        self.alpha = alpha

        if parameters is not None or grid is not None or method is not None or initial is not None or forcing is not None:
            self.setup(parameters, grid, method, initial, forcing, self.alpha)

    def setup(self, parameters, grid, method, initial, forcing, alpha=0):
        if parameters is None or not len(parameters) == 11:
            raise Exception("Incorrect number of parameters\n- required: (s, f0, beta, p, nu, r, kappa, epsilon, kf, dkf, sigma)")

        if grid is None or not len(grid) == 2:
            raise Exception("Incorrect size of grid\n -required: (N, M)")

        s, f0, beta, p, nu, r, kappa, epsilon, kf, dkf, sigma = parameters
        N, M = grid

        if not isinstance(method, str):
            raise Exception("method is not a string")
            
        if initial is None:
            raise Exception(f"Initial conditions is not definded")
        
        if forcing is None:
            raise Exception(f"Forcing function is not definded")

        if not isinstance(s, (int, float)):
            raise Exception("s is not a number")

        if not isinstance(f0, (int, float)):
            raise Exception("f0 is not a number")
        
        if not isinstance(beta, (int, float)):
            raise Exception("beta is not a number")

        if not isinstance(p, int):
            raise Exception("p is not a int number")

        if not isinstance(nu, (int, float)):
            raise Exception("nu is not a number")

        if not isinstance(r, (int, float)):
            raise Exception("r is not a number")

        if not isinstance(kappa, (int, float)):
            raise Exception("kappa is not a number")

        if not isinstance(epsilon, (int, float)):
            raise Exception("epsilon is not a number")

        if not isinstance(kf, (int, float)):
            raise Exception("kf is not a number")

        if not isinstance(dkf, (int, float)):
            raise Exception("dkf is not a number")

        if not isinstance(sigma, (int, float)):
            raise Exception("sigma is not a number")

        if not isinstance(alpha, (int, float)):
            raise Exception("alpha is not a number")

        if not isinstance(N, (int, float)):
            raise Exception("N in grid is not a number")

        if not isinstance(M, (int, float)):
            raise Exception("M in grid is not a number")

        if not (s == 1 or s == -1):
            raise Exception("s is not sign (+1 or -1)")

        if not (alpha >= 0 and alpha <= 1):
            raise Exception("alpha is not in range [0,1]")

        if not initial in self._initials:
            raise Exception(f"{initial} is not definded as a initial conditions\n- available: {list(self._initials.keys())}")

        if not forcing in self._forcings:
            raise Exception(f"{forcing} is not definded as a forcing function\n- available: {list(self._forcings.keys())}")

        self.parameters = parameters
        self.s, self.f0, self.beta = s, f0, beta
        self.p, self.r, self.nu, self.kappa = p, r, nu, kappa
        self.epsilon, self.kf, self.dkf, self.sigma = epsilon, kf, dkf, sigma

        self.grid = grid
        self.N, self.M = N, M
        self.method = method
        self._s_0 = self._initials[initial]
        self._xi_hat = self._forcings[forcing]

        self._grid(self.N)
        
        return self
        
    def calc(self, s_n):
        t_n, q_hat_n = s_n
        q_n = self._ifft_phys(q_hat_n)
        return (t_n, q_n)

    def start(self):
        return self._s_0(self.N)

    def model(self):
        self.dt = 1/self.M
        h = self.dt
        method = self.method
        def f(t, q_hat):
            right_part =  - self._J_adv(q_hat)
            right_part += - self.r*q_hat - self.nu * (self.k2)**(self.p) * q_hat
            right_part +=  self.epsilon * self.k2 * self._xi_hat(t, q_hat)
            return right_part
        return f, h, method

    def scale(self, alpha = None):
        if alpha is None:
            alpha = self.alpha
        L = alpha * R_planet
        T = Omega_planet 
        return L, T

    def optimal(self, alpha = None, grid = None):
        if alpha is None:
            alpha = self.alpha
        if grid is None:
            grid = self.grid
        r_m, beta_m = 1, 1
        return r_m, beta_m

    def _grid(self, N):
        # 1D
        self.l = 2 * PI * np.linspace(0, 1, N, endpoint=False)
        self.dl = (2*PI)/N
        self.k = 2 * PI * np.fft.fftfreq(N, d=(self.dl))
        self.dk = 1

        # 2D
        self.X, self.Y = np.meshgrid(self.l, self.l, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(self.k, self.k, indexing='ij')
        self.x, self.y =  self.l[:, None], self.l[None, :]
        self.kx, self.ky =  self.k[:, None], self.k[None, :]

        # additioal grid
        self.ikx, self.iky = 1j * self.kx,  1j * self.ky
        self.k2 = self.kx**2 + self.ky**2
        self.k1 = np.sqrt(self.k2)
        self.k_res = self.k2 == self.kappa**2
        self.resonans = np.where(self.k_res, np.ones((N,N)), np.zeros((N,N)))
        self.k_max = np.max(np.abs(self.k))
        
        # padding grid
        self.N_pad = int(3 * self.k_max)
        self.l_pad = 2 * PI * np.linspace(0, 1, self.N_pad, endpoint=False)
        self.dl_pad = (2*PI)/self.N_pad
        self.k_pad = 2 * PI * np.fft.fftfreq(self.N_pad, d=(self.dl_pad))
        self.Kx_pad, self.Ky_pad = np.meshgrid(self.k_pad, self.k_pad, indexing='ij')
        self.kx_pad, self.ky_pad =  self.k_pad[:, None], self.k_pad[None, :]
        self.ikx_pad, self.iky_pad = 1j * self.kx_pad,  1j * self.ky_pad
        self.k2_pad = self.kx_pad**2 + self.ky_pad**2
        self.k1_pad = np.sqrt(self.k2_pad)
        self.k_max_pad = np.max(np.abs(self.k_pad))

        # other parameters
        self.f_hat = self._f_cor_hat()

    def _fft_phys(self, x):
        return 1/self.N**2 * np.fft.fft2(x)

    def _ifft_phys(self, x_hat):
        return self.N**2 * np.real(np.fft.ifft2(x_hat))

    def _fft_pad(self, x):
        return 1/self.N_pad**2 * np.fft.fft2(x)

    def _ifft_pad(self, x_hat):
        return self.N_pad**2 * np.real(np.fft.ifft2(x_hat))

    def _f_cor_hat(self):
        f_hat = np.zeros((self.N, self.N), dtype=complex)
        f_hat[0,0] = self.s * self.f0 + self.beta * PI * (self.N-1)/(self.N)
        mask = np.logical_and(self.Kx == 0, self.Ky != 0)
        phi = self.Ky[mask] * (2*PI/self.N)
        f_hat[mask] = - (self.beta*PI)/(self.N) * (1 - 1j * np.sin(phi)/(1 - np.cos(phi)))
        return f_hat

    def _psi_hat(self, q_hat, eps = 0.01):
        psi_hat = (q_hat - self.f_hat) / (self.k2 - self.kappa**2 + 1j * eps * self.resonans)
        if self.kappa == 0:
            psi_hat[self.k_res] = np.real(psi_hat[self.k_res])
        return psi_hat

    def _u_hat(self, q_hat):
        psi_hat = self._psi_hat(q_hat)
        ux_hat, uy_hat = - self.iky * psi_hat, + self.ikx * psi_hat
        return (ux_hat, uy_hat)

    def _pad_spectrum(self, a):
        n_p = + (self.N+1)//2
        n_m = - (self.N)//2 
        a_pad = np.zeros((self.N_pad, self.N_pad), dtype=complex)

        a_pad[:n_p, :n_p] = a[:n_p, :n_p]
        a_pad[n_m:, n_m:] = a[n_m:, n_m:]
        a_pad[:n_p, n_m:] = a[:n_p, n_m:]
        a_pad[n_m:, :n_p] = a[n_m:, :n_p]

        return a_pad

    def _crop_spectrum(self, a_pad):
        n_p = + (self.N+1)//2
        n_m = - (self.N)//2 
        a = np.zeros((self.N, self.N), dtype=complex)

        a[:n_p, :n_p] = a_pad[:n_p, :n_p]
        a[n_m:, n_m:] = a_pad[n_m:, n_m:]
        a[:n_p, n_m:] = a_pad[:n_p, n_m:]
        a[n_m:, :n_p] = a_pad[n_m:, :n_p]

        return a

    def _J_adv(self, q_hat):
        ux_hat, uy_hat = self._u_hat(q_hat)
        dx_q_hat, dy_q_hat = self.ikx * q_hat, self.iky * q_hat

        ux_hat_pad, uy_hat_pad = self._pad_spectrum(ux_hat), self._pad_spectrum(uy_hat)
        dx_q_hat_pad, dy_q_hat_pad = self._pad_spectrum(dx_q_hat), self._pad_spectrum(dy_q_hat)
        ux_pad, uy_pad = self._ifft_pad(ux_hat_pad), self._ifft_pad(uy_hat_pad)
        dx_q_pad, dy_q_pad = self._ifft_pad(dx_q_hat_pad), self._ifft_pad(dy_q_hat_pad)

        j_adv_pad = ux_pad * dx_q_pad + uy_pad * dy_q_pad
        j_adv_hat_pad = self._fft_pad(j_adv_pad)
        j_adv_hat = self._crop_spectrum(j_adv_hat_pad)

        return j_adv_hat

    def _s_zero(self, N):
        q_0 = np.zeros((N, N))
        q_hat_0 = self._fft_phys(q_0)
        return (0, q_hat_0)

    def _s_monople(self, N, rho = 0.5, w=0.2):
        q_0, r0 = 0, PI
        R_n = np.sqrt((self.x - r0)**2 + (self.y - r0)**2)
        q_0 += 0.5 * (np.tanh((rho/L * r0 - R_n)/w) + 1)
        q_hat_0 = self._fft_phys(q_0)
        return (0, q_hat_0)

    def _s_dipole(self, N, rho = 0.5, w=0.2):
        q_0, r0 = 0, PI
        R_p = np.sqrt((self.x - r0 + r0/2)**2 + (self.y - r0)**2)
        R_m = np.sqrt((self.x - r0 - r0/2)**2 + (self.y - r0)**2)
        q_p = + 0.5 * (np.tanh((rho/2 * r0 - R_p)/w) + 1)
        q_m = - 0.5 * (np.tanh((rho/2 * r0 - R_m)/w) + 1)
        q_0 += q_p + q_m
        q_hat_0 = self._fft_phys(q_0)
        return (0, q_hat_0)

    def _s_random(self, N):
        q_0 = np.random.random((N, N))
        q_hat_0 = self._fft_phys(q_0)
        return (0, q_hat_0)

    def _xi_zero(self, t, q_hat):
        return np.zeros(q_hat.shape)

    def _xi_random(self, t, q_hat):
        xi = np.random.random(q_hat.shape)
        xi_hat = self._fft_phys(xi)
        return xi_hat

