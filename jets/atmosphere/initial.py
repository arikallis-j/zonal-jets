import jax.numpy as jnp
import jax.random as rnd

from .transform import fft_phys

def q_zero(params, grid):
    q_0 = jnp.zeros((grid.N, grid.N))
    q_hat_0 = fft_phys(q_0, grid)
    return q_hat_0

def q_random(params, grid):
    key = rnd.key(params.initial_seed)
    q_0 = 2*rnd.uniform(key, (grid.N, grid.N)) - 1
    q_0 -= q_0.mean()
    q_hat_0 = fft_phys(q_0, grid)
    print(q_hat_0)
    return q_hat_0

def q_random_vel(params, grid):
    key = rnd.key(params.initial_seed)
    key_1, key_2 = rnd.split(key)
    ux = rnd.uniform(key_1, (grid.N, grid.N))
    uy = rnd.uniform(key_2, (grid.N, grid.N))
    ux -= ux.mean()
    uy -= uy.mean() 
    ux_hat = fft_phys(ux, grid)
    uy_hat = fft_phys(uy, grid)
    q_hat_0 = grid.iKx * uy_hat - grid.iKy * ux_hat
    return q_hat_0

def q_monopole(params, grid):
    r0, rho, w = jnp.pi, 0.5, 0.2
    R_n = jnp.sqrt((grid.X - r0)**2 + (grid.Y - r0)**2)
    q_0 = 0.5 * (jnp.tanh((rho * r0 - R_n)/w) + 1)
    q_hat_0 = fft_phys(q_0, grid)
    return q_hat_0

def q_dipole(params, grid):
    r0, rho, w = jnp.pi, 0.5, 0.2
    R_p = jnp.sqrt((grid.X - r0 + r0/2)**2 + (grid.Y - r0)**2)
    R_m = jnp.sqrt((grid.X - r0 - r0/2)**2 + (grid.Y - r0)**2)
    q_p = + 0.5 * (jnp.tanh((rho/2 * r0 - R_p)/w) + 1)
    q_m = - 0.5 * (jnp.tanh((rho/2 * r0 - R_m)/w) + 1)
    q_0 = q_p + q_m
    q_hat_0 = fft_phys(q_0, grid)
    return q_hat_0

initials = {
    "zero": q_zero,
    "rnd": q_random,
    "rnd-vel": q_random_vel,
    "1-pole": q_monopole,
    "2-pole": q_dipole,
}