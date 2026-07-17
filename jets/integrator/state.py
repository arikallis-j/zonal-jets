import jax.numpy as jnp
from typing import NamedTuple

class State(NamedTuple):
    time: jnp.ndarray
    field: jnp.ndarray