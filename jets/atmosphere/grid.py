import jax.numpy as jnp
from typing import NamedTuple

class Grid(NamedTuple):
    N: int
    k_max: int
    k_bin_max: int
    N_pad: int
    dr: float
    dk: float
    r: jnp.ndarray
    k: jnp.ndarray
    k1: jnp.ndarray
    k_flat: jnp.ndarray
    X: jnp.ndarray
    Y: jnp.ndarray
    Kx: jnp.ndarray
    Ky: jnp.ndarray    
    iKx: jnp.ndarray
    iKy: jnp.ndarray
    K1: jnp.ndarray
    K2: jnp.ndarray


def make_grid(N):
    r = 2 * jnp.pi * jnp.linspace(0, 1, N, endpoint=False)
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=(2*jnp.pi/N))

    X, Y = jnp.meshgrid(r, r, indexing='ij')
    Kx, Ky = jnp.meshgrid(k, k, indexing='ij')
    iKx, iKy = 1j * Kx,  1j * Ky
    K2 = Kx**2 + Ky**2
    K1 = jnp.sqrt(K2)
    k1 = jnp.arange((jnp.floor(K1).astype(int)).max() + 1)
    k_bin = jnp.floor(K1).astype(jnp.int32)
    k_flat = jnp.ravel(k_bin)
    k_bin_max = len(k1)
    k_max = int(jnp.max(jnp.abs(k)))
    N_pad = 3 * k_max

    dr, dk = 2.0*jnp.pi/N, 1.0

    return Grid(N, k_max, k_bin_max, N_pad, dr, dk, r, k, k1, k_flat, X, Y, Kx, Ky, iKx, iKy, K1, K2)