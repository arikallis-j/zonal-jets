import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import json, os
from jax import lax, jit, random
from jax.numpy import pi as PI
key = random.key(42)

def fft2(x):
    return jnp.fft.fft2(x)

def ifft2(xhat):
    return jnp.fft.ifft2(xhat)

class Atmosphere:
    def __init__(self, N=256, h = 0.005, kf=32, dk=1, U = 10, L = 1e6, mu = 1e-8, nu = 1e-5, p = 1, beta = 1e-11, epsilon = 1e-10):
        self.U = U
        self.L = L
        self.p = p
        self.mu = mu * (L/U)
        self.nu = nu * 1/(L**(2*p-1)*U)
        self.beta = beta * (L**2/U)
        self.epsilon = epsilon * (L/U)**2
        
        self.l = 2 * PI
        self.N = N
        self.h = h
        self.kf = kf
        self.dk = dk
        self._init_grid()
        self._init_terms()
        self._init_calc()

    def _init_grid(self):
        x = jnp.linspace(0, self.l, self.N, endpoint=False)
        y = jnp.linspace(0, self.l, self.N, endpoint=False)
        self.X, self.Y = jnp.meshgrid(x, y, indexing='ij')

        os.makedirs(f"data", exist_ok=True)
        file_path = "data/zeta.json"
        if os.path.exists(file_path):
            print("Loading zeta...")
            file_path = "data/zeta.json"
            with open(file_path, 'r') as f:
                self.zeta = jnp.array(json.load(f))
            file_path = "data/T.json"
            with open(file_path, 'r') as f:
                self.t = json.load(f)
            print("Zeta loaded.")
        else:
            print("Creating zeta...")
            self.zeta = jnp.zeros((self.N, self.N))
            self.t = 0
            file_path = f"data/zeta.json"
            with open(file_path, 'w') as f:
                json.dump(self.zeta.tolist(), f, indent=4)
            file_path = f"data/T.json"
            with open(file_path, 'w') as f:
                json.dump(self.t, f, indent=4)
            print("Zeta created.")
        
        self.k =  2 * PI * jnp.fft.fftfreq(self.N, d=(self.l/self.N))
        self.kx = self.k[:, None]
        self.ky = self.k[None, :]
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        k2 = self.kx**2 + self.ky**2
        self.k2 = jnp.where(k2 == 0.0, 1.0, k2) 

        kxmax = jnp.max(jnp.abs(self.k))
        cutoff = 2.0/3.0 * kxmax
        Kx, Ky = jnp.meshgrid(jnp.abs(self.k), jnp.abs(self.k), indexing='ij')
        self.dealiasing = jnp.where((Kx < cutoff) & (Ky < cutoff), 1.0, 0.0)
        
        k = jnp.sqrt(k2)
        self.forcing = jnp.where((k >= (self.kf-self.dk)) & (k <= (self.kf+self.dk)), 1.0, 0.0)

    def _init_terms(self):
        self.a = - self.mu - (-1)**(self.p + 1) * self.nu * self.k2 + self.ikx/self.k2 * self.beta

        @jit
        def b(zeta):
            zeta_d = self.dealiasing * zeta
            u = jnp.real(ifft2(+ self.iky/self.k2 * zeta_d))
            v = jnp.real(ifft2(- self.ikx/self.k2 * zeta_d))
            dzeta_dx = jnp.real(ifft2(self.ikx * zeta_d))
            dzeta_dy = jnp.real(ifft2(self.iky * zeta_d))
            adv = u * dzeta_dx + v * dzeta_dy
            A = fft2(adv)

            key_r, key_i = random.split(key)
            real = random.normal(key_r, zeta_d.shape)
            imag = random.normal(key_i, zeta_d.shape)
            xi_tilde = self.dealiasing * self.forcing * (- self.k2 * (real + 1j * imag))
            xi = ifft2(xi_tilde)
            E = self.epsilon * fft2(xi - xi.mean())

            N = E - A

            return N
        
        self.b = b

        self.phi = jnp.exp(self.a * self.h/2)
        self.phi2 = self.phi**2

    def _init_calc(self):
        @jit
        def rk4(z):
            d1 = self.b(z)
            d2 = self.b((z + self.h/2*d1) * self.phi) / self.phi
            d3 = self.b((z + self.h/2*d2) * self.phi) / self.phi
            d4 = self.b((z + self.h*d3) * self.phi2) / self.phi2
            return self.phi2 * (z + self.h/6 * (d1 + 2*d2 + 2*d3 + d4))
        
        @jit
        def time_step(n, zeta_hat):
            return rk4(zeta_hat)

        @jit
        def integrate(zeta_hat_0, steps):
            return lax.fori_loop(0, steps, time_step, zeta_hat_0)

        self.integrate = integrate

    def calc(self, n=10):
        file_path = "data/zeta.json"
        with open(file_path, 'r') as f:
            self.zeta = jnp.array(json.load(f))
            
        file_path = "data/T.json"
        with open(file_path, 'r') as f:
            self.t = json.load(f)

        self.zeta_tilde = fft2(self.zeta)

        self.zeta_tilde = self.integrate(self.zeta_tilde, n)
        self.zeta = jnp.real(ifft2(self.zeta_tilde))
        self.ux =  jnp.real(ifft2(+self.iky/self.k2 * self.zeta_tilde * self.dealiasing))
        self.uy =  jnp.real(ifft2(-self.ikx/self.k2 * self.zeta_tilde * self.dealiasing))
        self.u = jnp.sqrt(self.ux**2 + self.uy**2)

        self.u_mean = jnp.mean(self.u)
        self.n_R = (self.beta/(2*self.u_mean))**(1/2)

        self.e_tot = self.u_mean**2/2
        self.t += self.h * n
        self.Epsilon = self.e_tot/self.t
        self.n_beta = 0.5 * (self.beta**3/self.Epsilon)**(1/5)
        
        self.R_beta = self.n_beta/self.n_R

        file_path = "data/zeta.json"
        with open(file_path, 'w') as f:
            json.dump(self.zeta.tolist(), f, indent=4)

        file_path = f"data/T.json"
        with open(file_path, 'w') as f:
            json.dump(self.t, f, indent=4)

    def plot_zeta(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.zeta), levels=60)
        plt.colorbar()
        plt.title("Vorticity")
        if save:
            plt.savefig('zeta.png')
        if show:
            plt.show()

    def plot_U(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.u), levels=60)
        plt.colorbar()
        plt.title("U(x,y)")
        if save:
            plt.savefig('U.png')
        if show:
            plt.show()
