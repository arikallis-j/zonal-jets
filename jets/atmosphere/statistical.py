import jax.numpy as jnp

from .transform import *
from .physical import *

def calc_energy_space(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    E = 0.5 * grid.K1**2 * (jnp.abs(psi_hat))**2
    return E

def calc_enstrophy_space(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    Z = 0.5 * (grid.K1)**4 * (jnp.abs(psi_hat))**2
    return Z

def calc_energy_spectrum(q_hat, params, grid):
    E = calc_energy_space(q_hat, params, grid)
    e = field_spectrum(E, grid)
    return e

def calc_enstrophy_spectrum(q_hat, params, grid):
    Z = calc_enstrophy_space(q_hat, params, grid)
    z = field_spectrum(Z, grid)
    return z

def calc_energy_flow(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    adv_hat = calc_adv_hat(q_hat, params, grid)
    T_e = - jnp.real(jnp.conj(psi_hat) * adv_hat)
    t_e = field_spectrum(T_e, grid)
    pi_e = - jnp.cumsum(t_e)
    return pi_e

def calc_enstrophy_flow(q_hat, params, grid):
    adv_hat = calc_adv_hat(q_hat, params, grid)
    T_z = - jnp.real(jnp.conj(q_hat) * adv_hat)
    t_z = field_spectrum(T_z, grid)
    pi_z = - jnp.cumsum(t_z)
    return pi_z