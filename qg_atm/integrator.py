"""Runge–Kutta methods
let: dy_n/dt_n = f(t_n, y_n) 
-->  y_n+1 = y_n + h * Sum_(i = 1)^S b_i k_i
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from functools import partial
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

class State(NamedTuple):
    t: jnp.ndarray
    y: jnp.ndarray

def make_state(t, y):
    return State(jnp.array(t), jnp.array(y))

def rk1(s, f, h):
    t0, y0 = s.t, s.y
    k1 = f(t0, y0)
    t1, y1 = t0 + h, y0 + h * k1
    return State(t1, y1)

def rk2(s, f, h):
    t0, y0 = s.t, s.y
    k1 = f(t0, y0)
    k2 = f(t0 + h, y0 + h * k1)
    t1, y1 = t0 + h, y0 + h/2 * (k1 + k2)
    return State(t1, y1)

def rk3(s, f, h):
    t0, y0 = s.t, s.y
    k1 = f(t0, y0)
    k2 = f(t0 + h/2, y0 + h/2 * k1)
    k3 = f(t0 + h, y0 - h * k1 + 2 * h * k2)
    t1, y1 = t0 + h, y0 + h/6 * (k1 + 4 * k2 + k3)
    return State(t1, y1)

def rk4(s, f, h):
    t0, y0 = s.t, s.y
    k1 = f(t0, y0)
    k2 = f(t0 + h/2, y0 + h/2 * k1)
    k3 = f(t0 + h/2, y0 + h/2 * k2)
    k4 = f(t0 + h, y0 + h * k3)
    t1, y1 = t0 + h, y0 + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return State(t1, y1)

@partial(jit, static_argnames=['step', 'n_steps'])
def jit_integrate(state, step, n_steps):
    return lax.fori_loop(0, n_steps, step, state)

class Integrator:
    """Integrator for equation:
        dy/dt = f(t, y)
    - f function : integrated function
    - h: step of integration
    - method: name of Runge–Kutta methods
    - n_steps: number of integration's step
    """
    def __init__(self, f=None, h=None, method=None, n_steps=1):
        self._methods = {
            'rk1': rk1,
            'rk2': rk2,
            'rk3': rk3,
            'rk4': rk4,
        }

        self._f_int, self._status, self._devices = None, None, None

        if f is not None or h is not None or method is not None:
            self.setup(f, h, method, n_steps)

    def setup(self, f, h, method, n_steps=1):
        if not callable(f):
            raise ValueError("f is not a function")

        if not isinstance(h, (int, float)):
            raise ValueError("h is not a number")

        if not isinstance(method, str):
            raise ValueError("method is not a string")

        if not method in self._methods:
            raise KeyError(f"{method} is not definded as a method")

        self.clear()

        step = lambda i, s: partial(self._methods[method], f=f, h=h)(s)
        self._f_int = lambda s: partial(jit_integrate, step=step, n_steps=n_steps)(s)
        self._devices = jax.devices()
        self._status = {'method': method, 'h': h,'n_steps': n_steps, 'devs': self._devices}

        return self

    def clear(self):
        self._f_int = None
        self._status = None
        self._devices = None

    def integrate(self, s):
        if self._f_int is not None:
            return self._f_int(s)
        else:
            raise Exception("Setup Integrator before using")

    def f_int(self):
        if self._f_int is not None:
            return self._f_int
        else:
            raise Exception("Setup Integrator before using")

    def status(self):
        return self._status

    def methods(self):
        return list(self._methods.keys())

    def devices(self):
        return self._devices

    def add_method(self, method, name):
        if method.__code__.co_argcount != 3:
            raise TypeError("method must have 3 arguments: g = g(s, f, h)")

        self._methods[name] = method
            
    def __str__(self):
        return f"methods: {list(self._methods.keys())}\nstatus: {self._status}"
