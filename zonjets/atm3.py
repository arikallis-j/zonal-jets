import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import xarray as xr
import os
from jax import lax, jit, random, devices
from jax.numpy import pi as PI

def fft2(x):
    return jnp.fft.fft2(x)

def ifft2(xhat):
    return jnp.fft.ifft2(xhat)