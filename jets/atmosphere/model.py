import jax.numpy as jnp
from typing import NamedTuple
import xarray as xr

from .physical import *
from .statistical import *
from .transform import ifft_phys

class phys_qgc_1(NamedTuple):
    t: jnp.ndarray
    q: jnp.ndarray
    psi: jnp.ndarray
    ux: jnp.ndarray
    uy: jnp.ndarray
    v: jnp.ndarray
    omega: jnp.ndarray
    r: jnp.ndarray

class stat_qgc_1(NamedTuple):
    t: jnp.ndarray
    E: jnp.ndarray
    Z: jnp.ndarray
    e: jnp.ndarray
    z: jnp.ndarray
    pi_e: jnp.ndarray
    pi_z: jnp.ndarray
    k: jnp.ndarray
    k1: jnp.ndarray

class nums_qgc_1(NamedTuple):
    t: jnp.ndarray
    e_mean: jnp.ndarray
    z_mean: jnp.ndarray
    e_dot: jnp.ndarray
    z_dot: jnp.ndarray
    R_beta: jnp.ndarray

class params_qgc_1(NamedTuple):
    initial: str = "zero"
    forcing: str = "zero"
    initial_seed: int = 42
    forcing_seed: int = 42
    p: int = 1
    beta: float = 0.0
    nu: float = 0.0
    r: float = 0.0
    kappa: float = 0.0
    epsilon: float = 0.0
    kf: float = 32.0
    dkf: float = 1.0
    sigma: float = 1.0
    forcing_dt: float = 0.01

def rhs_qgc_1(t, q_hat, params, grid):
    adv_hat = calc_adv_hat(q_hat, params, grid)
    cor_hat = calc_cor_hat(q_hat, params, grid)
    dif_hat = calc_dif_hat(q_hat, params, grid)
    xi_hat = calc_xi_hat(t, params, grid)
    rhs = - adv_hat - cor_hat + dif_hat + xi_hat
    return rhs

def atm_phys_qgc_1(t, q_hat, params, grid):
    psi_hat = calc_psi_hat(q_hat, params, grid)
    ux_hat, uy_hat = calc_u_hat(q_hat, params, grid)
    v = calc_velocity(q_hat, params, grid)
    omega = calc_vorticity(q_hat, params, grid)
    q, psi = ifft_phys(q_hat, grid), ifft_phys(psi_hat, grid)
    ux, uy = ifft_phys(ux_hat, grid), ifft_phys(uy_hat, grid)
    phys = phys_qgc_1(t, q, psi, ux, uy, v, omega, grid.r)
    return phys

def atm_stat_qgc_1(t, q_hat, params, grid):
    E = calc_energy_space(q_hat, params, grid)
    Z = calc_enstrophy_space(q_hat, params, grid)
    e = calc_energy_spectrum(q_hat, params, grid)
    z = calc_enstrophy_spectrum(q_hat, params, grid)
    pi_e = calc_energy_flow(q_hat, params, grid)
    pi_z = calc_enstrophy_flow(q_hat, params, grid)
    stat = stat_qgc_1(t, E, Z, e, z, pi_e, pi_z, grid.k, grid.k1)
    return stat

def atm_nums_qgc_1(t, q_hat, params, grid):
    v = calc_velocity(q_hat, params, grid)
    omega = calc_vorticity(q_hat, params, grid)
    e_mean, z_mean = field_mean_power(v), field_mean_power(omega)
    e_dot = calc_energy_injection(t, q_hat, params, grid)
    z_dot = calc_enstrophy_injection(t, q_hat, params, grid)
    R_beta = calc_R_beta(t, q_hat, params, grid)
    nums = nums_qgc_1(t, e_mean, z_mean, e_dot, z_dot, R_beta)
    return nums

def atm_dataset_qgc_1(t, q_hat, physics, statistics, numbers, descript={}):
    phys, stat, nums = physics(t, q_hat), statistics(t, q_hat), numbers(t, q_hat)
    phys, stat, nums = phys._asdict(), stat._asdict(), nums._asdict()
    data_vars, coords = {}, {}

    coords['x'], coords['y'] = phys['r'], phys['r']
    coords['kx'], coords['ky'] = stat['k'], stat['k']
    coords['k1'] = stat['k1']

    phys_2d = ['q', 'psi', 'ux', 'uy', 'v', 'omega']
    stat_2d = ['E', 'Z']
    stat_1d = ['e', 'z', 'pi_e', 'pi_z']
    nums_0d = ['e_mean', 'z_mean', 'e_dot', 'z_dot', 'R_beta']

    for key in phys_2d:
        data_vars[key] = (('x', 'y'), phys[key])
    for key in stat_2d:
        data_vars[key] = (('kx', 'ky'), stat[key])
    for key in stat_1d:
        data_vars[key] = (('k1', ), stat[key])
    for key in nums_0d:
        data_vars[key] = ((), nums[key])

    ds = xr.Dataset(data_vars = data_vars, coords = coords, attrs = descript)
    ds = ds.expand_dims(t=[phys['t']])

    return ds

def initial_qgc_1(params, grid):
    q_0_hat = calc_init_hat(params, grid)
    return 0.0, q_0_hat

models = {
    'qgc-1': {
        'meta' : {
            'model': 'qgc-1',
            'type' : "quasi-geostrophic",
            'coord' : "cartesian",
            'approach' : "spectral-1"
        },
        'params': params_qgc_1,
        'atm_phys': atm_phys_qgc_1,
        'atm_stat': atm_stat_qgc_1,
        'atm_nums': atm_nums_qgc_1,
        'atm_dataset': atm_dataset_qgc_1,
        'init_state': initial_qgc_1,
        'rhs': rhs_qgc_1,
    }
}