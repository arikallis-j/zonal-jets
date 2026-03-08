"""Runge–Kutta methods
let: dy_n/dt_n = f(t_n, y_n) 
-->  y_n+1 = y_n + h * Sum_(i = 1)^S b_i k_i
"""

class Integrator:
    """Description"""
    def __init__(self, f=None, h=None, method=None, N=1):
        self._methods = {
            'rk1': self._rk1,
            'rk2': self._rk2,
            'rk3': self._rk3,
            'rk4': self._rk4,
        }

        self._f_int, self._status = None, None

        if f is not None or h is not None or method is not None:
            self.setup(f, h, method, N)

    def add_method(self, func, name):
        self._methods[name] = func

    def setup(self, f, h, method, N=1):
        if not callable(f):
            raise ValueError("f is not a function")

        if not isinstance(h, (int, float)):
            raise ValueError("h is not a number")

        if not isinstance(method, str):
            raise ValueError("method is not a string")
        
        if f.__code__.co_argcount != 2:
            raise TypeError("f must have 2 arguments: f = f(t, y)")

        if not method in self._methods:
            raise KeyError(f"{method} is not definded as a method")

        self.clear()

        def f_int(s_n):
            s_k = s_n
            for k in range(N):
                s_k = self._methods[method](s_k, f, h)
            return s_k

        self._f_int, self._status = f_int, {'method': method, 'h': h,'N': N}

        return self._f_int

    def clear(self):
        self._f_int = None
        self._status = None

    def integrate(self, s_n):
        if callable(self._f_int):
            return self._f_int(s_n)
        else:
            raise Exception("Setup Integrator before using")

    def f_int(self):
        if callable(self._f_int):
            return self._f_int
        else:
            raise Exception("Setup Integrator before using")

    def status(self):
        return self._status

    def methods(self):
        return self._methods
            
    def __str__(self):
        return f"methods: {list(self._methods.keys())}\nstatus: {self._status}"

    def _rk1(self, s_n, f, h):
        """RK1: S = 1 
        b_1 = 1, k_1 = f(t_n, y_n)
        y_n+1 = y_n + h * k_1
        """
        t_n, y_n = s_n
        k_1 = f(t_n, y_n)
        t_n1, y_n1 = t_n + h, y_n + h * k_1
        return (t_n1, y_n1)

    def _rk2(self, s_n, f, h):
        """RK2: S=2 
        b_1 = 1/2, k_1 = f(t_n, y_n)
        b_2 = 1/2, k_2 = f(t_n + h, y_n + h * k_1)
        y_n+1 = y_n + h/2 * (k_1 + k_2)
        """
        t_n, y_n = s_n
        k_1 = f(t_n, y_n)
        k_2 = f(t_n + h, y_n + h * k_1)
        t_n1, y_n1 = t_n + h, y_n + h/2 * (k_1 + k_2)
        return (t_n1, y_n1)

    def _rk3(self, s_n, f, h):
        """RK3: S = 3:
        b_1 = 1/6, k_1 = f(t_n, y_n)
        b_2 = 4/6, k_2 = f(t_n + h/2, y_n + h/2 * k_1)
        b_3 = 1/6, k_3 = f(t_n + h, y_n - h * k_1 + 2 * h * k_2)
        y_n+1 = y_n + h/6 * (k_1 + 4 * k_2 + k_3)
        """
        t_n, y_n = s_n
        k_1 = f(t_n, y_n)
        k_2 = f(t_n + h/2, y_n + h/2 * k_1)
        k_3 = f(t_n + h, y_n - h * k_1 + 2 * h * k_2)
        t_n1, y_n1 = t_n + h, y_n + h/6 * (k_1 + 4 * k_2 + k_3)
        return (t_n1, y_n1)

    def _rk4(self, s_n, f, h):
        """RK4: S = 4:
        b_1 = 1/6, k_1 = f(t_n, y_n)
        b_2 = 2/6, k_2 = f(t_n + h/2, y_n + h/2 * k_1)
        b_3 = 2/6, k_3 = f(t_n + h/2, y_n + h/2 * k_2)
        b_4 = 1/6, k_4 = f(t_n + h, y_n + h * k_3)
        y_n+1 = y_n + h/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        """
        t_n, y_n = s_n
        k_1 = f(t_n, y_n)
        k_2 = f(t_n + h/2, y_n + h/2 * k_1)
        k_3 = f(t_n + h/2, y_n + h/2 * k_2)
        k_4 = f(t_n + h, y_n + h * k_3)
        t_n1, y_n1 = t_n + h, y_n + h/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        return (t_n1, y_n1)
