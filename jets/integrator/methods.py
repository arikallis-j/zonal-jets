from .state import State

def rk1(state, rhs, step):
    """Runge–Kutta 1st order method"""
    t0, y0, h = state.time, state.field, step
    k1 = rhs(t0, y0)
    t1, y1 = t0 + h, y0 + h * k1
    return State(t1, y1)

def rk2(state, rhs, step):
    """Runge–Kutta 2nd order method"""
    t0, y0, h = state.time, state.field, step
    k1 = rhs(t0, y0)
    k2 = rhs(t0 + h, y0 + h * k1)
    t1, y1 = t0 + h, y0 + h/2 * (k1 + k2)
    return State(t1, y1)

def rk3(state, rhs, step):
    """Runge–Kutta 3rd order method"""
    t0, y0, h = state.time, state.field, step
    k1 = rhs(t0, y0)
    k2 = rhs(t0 + h/2, y0 + h/2 * k1)
    k3 = rhs(t0 + h, y0 - h * k1 + 2 * h * k2)
    t1, y1 = t0 + h, y0 + h/6 * (k1 + 4 * k2 + k3)
    return State(t1, y1)

def rk4(state, rhs, step):
    """Runge–Kutta 4th order method"""
    t0, y0, h = state.time, state.field, step
    k1 = rhs(t0, y0)
    k2 = rhs(t0 + h/2, y0 + h/2 * k1)
    k3 = rhs(t0 + h/2, y0 + h/2 * k2)
    k4 = rhs(t0 + h, y0 + h * k3)
    t1, y1 = t0 + h, y0 + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return State(t1, y1)

methods = {
    'rk1': rk1,
    'rk2': rk2,
    'rk3': rk3,
    'rk4': rk4,
}

if __name__ == '__main__':
    print(methods)
    t = 1.0
    q = jnp.ones((5,5))
    s = State(t, q)
    h = 0.1
    rhs = lambda t, q: jnp.exp(-1/2*t) * q
    s1 = rk4(s, rhs, h)
    print(s1)