import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from .methods import methods
from .state import State

jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnames=['calc', 'n_points'])
def jit_integrate(state, calc, n_points):
    return lax.fori_loop(0, n_points, calc, state)

class Integrator:
    def __init__(self, rhs=None, method='rk4', interval=1, n_points=1):
        self._clear()

        if rhs is not None:
            self.setup(rhs, method, interval, n_points)

    def setup(self, rhs, method='rk4', interval=1, n_points=1):
        self._clear()
        self.step = interval/n_points
        calc = lambda k, state: partial(methods[method], rhs=rhs, step=self.step)(state)
        self._integral = lambda state: partial(jit_integrate, calc=calc, n_points=n_points)(state)
        
        self.status = {
            'method': method,
            'interval': interval,
            'n_points': n_points,
        }

    def integrate(self, state):
        if self.status is not None:
            return self._integral(state)
        else:
            raise Exception("Setup Integrator before using")

    def integral(self):
        return self._integral

    def methods(self):
        return list(methods.keys())

    def make_state(self, time, field):
        return State(jnp.asarray(time), jnp.asarray(field))

    def devices(self):
        return jax.devices()

    def _clear(self):
        self.status = None
        self._integral = None
        