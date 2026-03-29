import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, lax
import numpy as np
from functools import partial
from typing import NamedTuple
jax.config.update("jax_enable_x64", True)

class PhysicalState(NamedTuple):
    q: np.ndarray
    psi: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    X: np.ndarray
    Y: np.ndarray

class SpectralState(NamedTuple):
    q_hat: np.ndarray
    psi_hat: np.ndarray
    ux_hat: np.ndarray
    uy_hat: np.ndarray
    Kx: np.ndarray
    Ky: np.ndarray

class AtmosphereState(NamedTuple):
    alpha: np.ndarray
    t: np.ndarray
    p: PhysicalState
    s: SpectralState

class Scale(NamedTuple):
    N: int
    M: int

class Grid(NamedTuple):
    N: int
    M: int
    k_max: int
    N_pad: int
    dr: float
    dk: float
    dt: float
    r: jnp.ndarray
    k: jnp.ndarray
    X: jnp.ndarray
    Y: jnp.ndarray
    Kx: jnp.ndarray
    Ky: jnp.ndarray    
    iKx: jnp.ndarray
    iKy: jnp.ndarray
    K1: jnp.ndarray
    K2: jnp.ndarray

class Params(NamedTuple):
    s: int
    f0: float
    beta: float
    p: int 
    nu: float 
    r: float 
    kappa: float
    epsilon: float
    kf: float
    dkf: float
    sigma: float
    initial_seed: int
    forcing_seed: int


def fft_phys(x, grid):
    return 1/grid.N**2 * jnp.fft.fft2(x)

def ifft_phys(x_hat, grid):
    return grid.N**2 * jnp.real(jnp.fft.ifft2(x_hat))

def fft_pad(x, grid):
    return 1/grid.N_pad**2 * jnp.fft.fft2(x)

def ifft_pad(x_hat, grid):
    return grid.N_pad**2 * jnp.real(jnp.fft.ifft2(x_hat))

def pad_spectrum(a, grid):
    n_p = + (grid.N+1)//2
    n_m = - (grid.N)//2 
    a_pad = jnp.zeros((grid.N_pad, grid.N_pad), dtype=jnp.complex128)

    a_pad = a_pad.at[:n_p, :n_p].set(a[:n_p, :n_p])
    a_pad = a_pad.at[n_m:, n_m:].set(a[n_m:, n_m:])
    a_pad = a_pad.at[:n_p, n_m:].set(a[:n_p, n_m:])
    a_pad = a_pad.at[n_m:, :n_p].set(a[n_m:, :n_p])

    return a_pad

def crop_spectrum(a_pad, grid):
    n_p = + (grid.N+1)//2
    n_m = - (grid.N)//2 
    a = jnp.zeros((grid.N, grid.N), dtype=jnp.complex128)

    a = a.at[:n_p, :n_p].set(a_pad[:n_p, :n_p])
    a = a.at[n_m:, n_m:].set(a_pad[n_m:, n_m:])
    a = a.at[:n_p, n_m:].set(a_pad[:n_p, n_m:])
    a = a.at[n_m:, :n_p].set(a_pad[n_m:, :n_p])

    return a

def make_scale(N, M):
    return Scale(N, M)

def make_grid(scale):
    r = 2 * jnp.pi * jnp.linspace(0, 1, scale.N, endpoint=False)
    k = 2 * jnp.pi * jnp.fft.fftfreq(scale.N, d=(2*jnp.pi/scale.N))

    X, Y = jnp.meshgrid(r, r, indexing='ij')
    Kx, Ky = jnp.meshgrid(k, k, indexing='ij')

    iKx, iKy = 1j * Kx,  1j * Ky
    K2 = Kx**2 + Ky**2
    K1 = jnp.sqrt(K2)
    k_max = int(jnp.max(jnp.abs(k)))
    N_pad = 3 * k_max

    dr, dk, dt = 2.0*jnp.pi/scale.N, 1.0, 1.0/scale.M

    return Grid(scale.N, scale.M, k_max, N_pad, dr, dk, dt, r, k, X, Y, Kx, Ky, iKx, iKy, K1, K2)

def make_params(s=1, f0=1, beta=0, kappa=0, p=1, nu=0, r=0, epsilon=0, kf=0, dkf=0, sigma=0, initial_seed=42, forcing_seed=43):
    if not (s == 1 or s == -1):
        raise Exception("s is not sign (+1 or -1)")

    return Params(s, f0, beta, p, nu, r, kappa, epsilon, kf, dkf, sigma, initial_seed, forcing_seed)

def make_physical_state(q, psi, ux, uy, X, Y):
    return PhysicalState(
        np.array(q), 
        np.array(psi), 
        np.array(ux), 
        np.array(uy), 
        np.array(X),
        np.array(Y),
    )

def make_spectral_state(q_hat, psi_hat, ux_hat, uy_hat, Kx, Ky):
    return SpectralState(
        np.array(q_hat), 
        np.array(psi_hat), 
        np.array(ux_hat), 
        np.array(uy_hat), 
        np.array(Kx),
        np.array(Ky),
    )
        
def make_atmosphere_state(alpha, t, physical, spectral):
    return AtmosphereState(
        np.array(alpha),
        np.array(t),
        physical,
        spectral,
    )

def calc_f_cor_hat(params, grid):
    f_hat = jnp.zeros((grid.N, grid.N), dtype=jnp.complex128)
    f_hat = f_hat.at[0,0].set(params.s * params.f0 + params.beta * jnp.pi * (grid.N-1)/(grid.N))
    mask = jnp.logical_and(grid.Kx == 0, grid.Ky != 0)
    phi = grid.Ky[mask] * (2*jnp.pi/grid.N)
    f_hat = f_hat.at[mask].set(-(params.beta*jnp.pi)/(grid.N) * (1 - 1j * jnp.sin(phi)/(1 - jnp.cos(phi))))
    return f_hat

def calc_psi_hat(q_hat, f_hat, params, grid):
    psi_hat = - (q_hat - f_hat) / (grid.K2 + params.kappa**2)
    psi_hat_res = psi_hat.at[0,0].set(0.0)
    psi_hat = lax.cond(params.kappa == 0, lambda psi_hat: psi_hat_res, lambda psi_hat: psi_hat, psi_hat)
    return psi_hat

def calc_u_hat(psi_hat, grid):
    ux_hat, uy_hat = - grid.iKy * psi_hat, + grid.iKx * psi_hat
    return ux_hat, uy_hat

def calc_J_adv(q_hat, psi_hat, grid):
    ux_hat, uy_hat = calc_u_hat(psi_hat, grid)
    dx_q_hat, dy_q_hat = grid.iKx * q_hat, grid.iKy * q_hat

    ux_hat_pad, uy_hat_pad = pad_spectrum(ux_hat, grid), pad_spectrum(uy_hat, grid)
    dx_q_hat_pad, dy_q_hat_pad = pad_spectrum(dx_q_hat, grid), pad_spectrum(dy_q_hat, grid)
    ux_pad, uy_pad = ifft_pad(ux_hat_pad, grid), ifft_pad(uy_hat_pad, grid)
    dx_q_pad, dy_q_pad = ifft_pad(dx_q_hat_pad, grid), ifft_pad(dy_q_hat_pad, grid)

    j_adv_pad = ux_pad * dx_q_pad + uy_pad * dy_q_pad
    j_adv_hat_pad = fft_pad(j_adv_pad, grid)
    j_adv_hat = crop_spectrum(j_adv_hat_pad, grid)

    return j_adv_hat

def rhs_qg_model(t, q_hat, f_hat, xi_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, f_hat, params, grid)
    rhs = jnp.zeros((grid.N, grid.N))
    rhs += - calc_J_adv(q_hat, psi_hat, grid)
    rhs += - params.r*q_hat - params.nu * (grid.K2)**(params.p) * q_hat
    rhs += params.epsilon * grid.K2 * xi_hat(t, q_hat, params, grid)
    return rhs

def s_zero(params, grid):
    q_0 = jnp.zeros((grid.N, grid.N))
    q_hat_0 = fft_phys(q_0, grid)
    return (jnp.array(0.0), q_hat_0)

def s_monople(params, grid, rho = 0.5, w = 0.2):
    q_0, r0 = 0, jnp.pi
    R_n = jnp.sqrt((grid.X - r0)**2 + (grid.Y - r0)**2)
    q_0 += 0.5 * (jnp.tanh((rho * r0 - R_n)/w) + 1)
    q_hat_0 = fft_phys(q_0, grid)
    return (jnp.array(0.0), q_hat_0)

def s_dipole(params, grid, rho = 0.5, w = 0.2):
    q_0, r0 = 0, jnp.pi
    R_p = jnp.sqrt((grid.X - r0 + r0/2)**2 + (grid.Y - r0)**2)
    R_m = jnp.sqrt((grid.X - r0 - r0/2)**2 + (grid.Y - r0)**2)
    q_p = + 0.5 * (jnp.tanh((rho/2 * r0 - R_p)/w) + 1)
    q_m = - 0.5 * (jnp.tanh((rho/2 * r0 - R_m)/w) + 1)
    q_0 += q_p + q_m
    q_hat_0 = fft_phys(q_0, grid)
    return (jnp.array(0.0), q_hat_0)

def s_random(params, grid):
    key_state = rnd.key(params.initial_seed)
    q_0 = 2*rnd.uniform(key_state, (grid.N, grid.N)) - 1
    q_hat_0 = fft_phys(q_0, grid)
    return (jnp.array(0.0), q_hat_0)

def s_u_random(params, grid):
    key_state = rnd.key(params.initial_seed)
    key_1, key_2 = rnd.split(key_state)
    ux = rnd.uniform(key_state, (grid.N, grid.N))
    uy = rnd.uniform(key_state, (grid.N, grid.N))
    ux -= ux.mean()
    uy -= uy.mean() 
    ux_hat = fft_phys(ux, grid)
    uy_hat = fft_phys(uy, grid)
    omega_hat = grid.iKx * uy_hat - grid.iKy * ux_hat
    q_hat_0 = omega_hat * jnp.where(grid.K2 != 0, (grid.K2 + params.kappa**2)/grid.K2, jnp.ones((grid.N, grid.N)))
    return (jnp.array(0.0), q_hat_0)

def xi_zero(t, q_hat, params, grid):
    return jnp.zeros((grid.N, grid.N))

def xi_random(t, q_hat, params, grid):
    key, theta = rnd.key(params.forcing_seed), int(2*t/grid.dt)
    step_key = rnd.fold_in(key, theta)
    xi = rnd.uniform(step_key, (grid.N, grid.N))
    xi_hat = fft_phys(xi, grid)
    return xi_hat

class Atmosphere:
    def __init__(self, params = None, scale = None, method = None, initial = None, forcing = None, alpha = 0):
        self._initials = {
            "zero": s_zero,
            "monopole": s_monople,
            "dipole": s_dipole,
            "random": s_random,
            "u-random": s_u_random,
        }

        self._forcings = {
            "zero": xi_zero,
            "random": xi_random,
        }

        self.alpha = alpha

        if params is not None or scale is not None or method is not None or initial is not None or forcing is not None:
            self.setup(params, scale, method, initial, forcing, self.alpha)

    def setup(self, params, scale, method, initial, forcing, alpha=0):
        if not (alpha >= 0 and alpha <= 1):
            raise Exception("alpha is not in range [0,1]")

        if not initial in self._initials:
            raise Exception(f"{initial} is not definded as a initial conditions\n- available: {list(self._initials.keys())}")

        if not forcing in self._forcings:
            raise Exception(f"{forcing} is not definded as a forcing function\n- available: {list(self._forcings.keys())}")

        self.params = params
        self.scale = scale
        self.method = method
        self.initial = initial
        self.forcing = forcing
        self.alpha = alpha

        self.grid = make_grid(scale)
        self.f_hat = calc_f_cor_hat(self.params, self.grid)
        self.rhs = jit(lambda t, q_hat: rhs_qg_model(t, q_hat, self.f_hat, self._forcings[forcing], self.params, self.grid))
        
        return self

    def calc(self, state):
        t, q_hat = state.t, state.y
        psi_hat = calc_psi_hat(q_hat, self.f_hat, self.params, self.grid)
        ux_hat, uy_hat = calc_u_hat(psi_hat, self.grid)
        q, psi = ifft_phys(q_hat, self.grid), ifft_phys(psi_hat, self.grid)
        ux, uy = ifft_phys(ux_hat, self.grid), ifft_phys(uy_hat, self.grid)
        physical_state = make_physical_state(q, psi, ux, uy, self.grid.X, self.grid.Y)
        spectral_state = make_spectral_state(q_hat, psi_hat, ux_hat, uy_hat, self.grid.Kx, self.grid.Ky)
        atmosphere_state = make_atmosphere_state(self.alpha, t, physical_state, spectral_state)
        return atmosphere_state

    def start(self, **kwargs):
        return self._initials[self.initial](self.params, self.grid, **kwargs)

    def model(self):
        return self.rhs, self.grid.dt, self.method

    def scale(self, alpha = None):
        if alpha is None:
            alpha = self.alpha
        L = alpha * R_planet
        T = Omega_planet 
        return L, T

    def optimal(self, alpha = None, scale = None):
        if alpha is None:
            alpha = self.alpha
        if scale is None:
            scale = self.scale
        return None