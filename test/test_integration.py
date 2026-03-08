import numpy as np
from qg_atm import Integrator

def test_all():
    test_RKS()
    test_integrator()
    test_sin()


def test_RKS(N=1000):
    h = 0.1
    r = 0.5
    f_0 = lambda t, x: - r * x
    s_0 = (0, 2) 

    Int = Integrator()

    s_k, p_k, q_k, r_k = s_0, s_0, s_0, s_0
    for k in range(N):
        s_k = Int.methods()["rk1"](s_k, f_0, h)
        p_k = Int.methods()["rk2"](p_k, f_0, h)
        q_k = Int.methods()["rk3"](q_k, f_0, h)
        r_k = Int.methods()["rk4"](r_k, f_0, h)
        if k%(N//10) == 0:
            print(f"RK1: s_{k} = {s_k}")
            print(f"RK2: s_{k} = {p_k}")
            print(f"RK3: s_{k} = {q_k}")
            print(f"RK4: s_{k} = {r_k}")

def test_integrator(N = 100):
    h = 0.1
    r = 0.5
    f_0 = lambda t, x: - r * x
    s_0 = (0, 2) 
    f_int = Integrator().setup(f_0, h, "rk4", N=N)
    s_N = f_int(s_0)
    print(f"RK4: s_{N} = {s_N}")

def test_sin(N = 100):
    h = 0.1
    omega = 2 * np.pi / 5
    f_0 = lambda t, x: np.cos(t*omega)
    s_0 = (0, 0)
    Int = Integrator(f_0, h, "rk4", N=N)
    s_est = Int.integrate(s_0)
    s_true = s_est[0], np.sin(s_est[0]*omega)/omega
    print(f"s_rk4: s_{N} = {s_est}")
    print(f"s_true: s_{N} = {s_true}")

