import jax.numpy as jnp
from jax import ops

def fft(x, grid):
    return jnp.fft.fft2(x)

def ifft(x_hat, grid):
    return jnp.real(jnp.fft.ifft2(x_hat))

def fft_phys(x, grid):
    return 1/grid.N**2 * jnp.fft.fft2(x)

def ifft_phys(x_hat, grid):
    return grid.N**2 * jnp.real(jnp.fft.ifft2(x_hat))

def fft_pad(x, grid):
    return 1/grid.N_pad**2 * jnp.fft.fft2(x)

def ifft_pad(x_hat, grid):
    return grid.N_pad**2 * jnp.real(jnp.fft.ifft2(x_hat))

def enforce_hermitian(x_hat):
    x_flip = jnp.roll(jnp.roll(x_hat[::-1, ::-1], 1, axis=0), 1, axis=1)
    return 0.5 * (x_hat + jnp.conj(x_flip))

def pad_spectrum(x_hat, grid):
    n_p = + (grid.N+1)//2
    n_m = - (grid.N)//2 
    x_hat_pad = jnp.zeros((grid.N_pad, grid.N_pad), dtype=jnp.complex128)

    x_hat_pad = x_hat_pad.at[:n_p, :n_p].set(x_hat[:n_p, :n_p])
    x_hat_pad = x_hat_pad.at[n_m:, n_m:].set(x_hat[n_m:, n_m:])
    x_hat_pad = x_hat_pad.at[:n_p, n_m:].set(x_hat[:n_p, n_m:])
    x_hat_pad = x_hat_pad.at[n_m:, :n_p].set(x_hat[n_m:, :n_p])

    return x_hat_pad

def crop_spectrum(x_hat_pad, grid):
    n_p = + (grid.N+1)//2
    n_m = - (grid.N)//2 
    x_hat = jnp.zeros((grid.N, grid.N), dtype=jnp.complex128)

    x_hat = x_hat.at[:n_p, :n_p].set(x_hat_pad[:n_p, :n_p])
    x_hat = x_hat.at[n_m:, n_m:].set(x_hat_pad[n_m:, n_m:])
    x_hat = x_hat.at[:n_p, n_m:].set(x_hat_pad[:n_p, n_m:])
    x_hat = x_hat.at[n_m:, :n_p].set(x_hat_pad[n_m:, :n_p])

    return x_hat

def field_mean(x):
    return jnp.mean(x)

def field_rms(x):
    return jnp.sqrt(jnp.mean(jnp.square(x)))

def field_power(x):
    return 0.5 * jnp.square(x)

def field_mean_power(x):
    return field_mean(field_power(x))

def field_spectrum(x_hat, grid):
    x_hat_flat = jnp.ravel(x_hat)
    x_spec = ops.segment_sum(data=x_hat_flat, segment_ids=grid.k_flat, num_segments=grid.k_bin_max)
    return x_spec
