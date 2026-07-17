import jax.numpy as jnp
from jax import lax

from .transform import *
from .forcing import forcings
from .initial import initials

def calc_psi_hat(q_hat, params, grid):
    psi_hat = - q_hat / (grid.K2 + params.kappa**2)
    psi_hat_res = psi_hat.at[0,0].set(0.0)
    psi_hat = lax.cond(params.kappa == 0, lambda psi_hat: psi_hat_res, lambda psi_hat: psi_hat, psi_hat)
    return psi_hat

def calc_u_hat(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    ux_hat, uy_hat = - grid.iKy * psi_hat, + grid.iKx * psi_hat
    return ux_hat, uy_hat

def calc_cor_hat(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    cor_hat = params.beta * grid.iKx * psi_hat
    return cor_hat

def calc_dif_hat(q_hat, params, grid):
    dif_hat = - params.r*q_hat - params.nu * (grid.K2)**(params.p) * q_hat
    return dif_hat

def calc_adv_hat(q_hat, params, grid):
    ux_hat, uy_hat = calc_u_hat(q_hat, params, grid)
    dx_q_hat, dy_q_hat = grid.iKx * q_hat, grid.iKy * q_hat
    ux_hat_pad, uy_hat_pad = pad_spectrum(ux_hat, grid), pad_spectrum(uy_hat, grid)
    dx_q_hat_pad, dy_q_hat_pad = pad_spectrum(dx_q_hat, grid), pad_spectrum(dy_q_hat, grid)
    ux_pad, uy_pad = ifft_pad(ux_hat_pad, grid), ifft_pad(uy_hat_pad, grid)
    dx_q_pad, dy_q_pad = ifft_pad(dx_q_hat_pad, grid), ifft_pad(dy_q_hat_pad, grid)
    adv_pad = ux_pad * dx_q_pad + uy_pad * dy_q_pad
    adv_hat_pad = fft_pad(adv_pad, grid)
    adv_hat = crop_spectrum(adv_hat_pad, grid)
    return adv_hat

def calc_xi_hat(t, params, grid):
    xi_norm = forcings[params.forcing]
    xi_hat =  params.epsilon * xi_norm(t, params, grid)
    return xi_hat

def calc_init_hat(params, grid):
    q_hat = initials[params.initial](params, grid)
    return q_hat

def calc_velocity(q_hat, params, grid):
    ux_hat, uy_hat = calc_u_hat(q_hat, params, grid)
    ux, uy = ifft_phys(ux_hat, grid), ifft_phys(uy_hat, grid)
    v = jnp.sqrt(ux**2 + uy**2)
    return v

def calc_vorticity(q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    omega_hat = - grid.K2 * psi_hat
    omega = ifft_phys(omega_hat, grid)
    return omega

def calc_R_beta(t, q_hat, params, grid):
    beta = params.beta
    v = calc_velocity(q_hat, params, grid)
    v_mean = field_mean(v)
    e_tot = field_mean(field_power(v))
    Epsilon = e_tot/t
    n_R = (beta/(2*v_mean))**(1/2)
    n_beta = 0.5 * (beta**3/Epsilon)**(1/5)
    R_beta = n_beta/n_R
    return R_beta

def calc_energy_injection(t, q_hat, params, grid):
    F_hat = calc_xi_hat(t, params, grid)
    k2 = jnp.where(grid.K2 == 0, 1.0, grid.K2)
    e_dot = jnp.sum(jnp.real(jnp.conj(q_hat) * F_hat) / k2)
    return e_dot

def calc_enstrophy_injection(t, q_hat, params, grid):
    F_hat = calc_xi_hat(t, params, grid)
    z_dot = jnp.sum(jnp.real(jnp.conj(q_hat) * F_hat))
    return z_dot