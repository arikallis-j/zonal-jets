import jax.numpy as jnp
import jax.random as rnd

from .transform import fft_phys, enforce_hermitian

def xi_zero(t, params, grid):
    xi = jnp.zeros((grid.N, grid.N))
    xi_hat = fft_phys(xi, grid)
    return xi_hat

def xi_random(t, params, grid):
    key = rnd.key(params.forcing_seed)
    theta = jnp.floor(t / params.forcing_dt).astype(jnp.int32)
    step_key = rnd.fold_in(key, theta)
    xi = rnd.normal(step_key, (grid.N, grid.N))
    
    xi_hat = fft_phys(xi, grid)
    xi_hat = xi_hat.at[0, 0].set(0.0)

    xi_hat /= jnp.sqrt(jnp.sum(jnp.abs(xi_hat)**2))
    xi_hat /= jnp.sqrt(params.forcing_dt)

    return xi_hat

def xi_normal(t, params, grid):
    key = rnd.key(params.forcing_seed)
    theta = jnp.floor(t / params.forcing_dt).astype(jnp.int32)
    step_key = rnd.fold_in(key, theta)
    key_r, key_i = rnd.split(step_key)
    real = rnd.normal(key_r, (grid.N, grid.N))
    imag = rnd.normal(key_i, (grid.N, grid.N))
    xi_hat = real + 1j * imag

    mask = jnp.exp(-((grid.K1 - params.kf)**2) / (2 * params.dkf**2))
    xi_hat *= mask
    xi_hat = enforce_hermitian(xi_hat)
    xi_hat = xi_hat.at[0, 0].set(0.0)
    
    xi_hat /= jnp.sqrt(jnp.sum(jnp.abs(xi_hat)**2))
    xi_hat /= jnp.sqrt(params.forcing_dt)
    
    return xi_hat

forcings = {
    "zero": xi_zero,
    "rnd": xi_random,
    "norm": xi_normal,
}
