import jax.numpy as jnp
from qg_atm import Integrator, make_state

def test_all(N = 100):
    print("\n# test RKS: #")
    test_RKS(N)
    print("# test RKS: #\n")
    print("\n# test integrator: #")
    test_integrator(N)
    print("# test integrator: #\n")
    print("\n# test sin:#")
    test_sin(N)
    print("# test sin: #\n")


def test_RKS(N=100):
    h = 0.1
    r = 0.5
    f_0 = lambda t, x: - r * x
    s_0 = make_state(0, 2) 

    Int = Integrator()

    s_k, p_k, q_k, r_k = s_0, s_0, s_0, s_0
    for k in range(N):
        s_k = Int.methods()["rk1"](s_k, f_0, h)
        p_k = Int.methods()["rk2"](p_k, f_0, h)
        q_k = Int.methods()["rk3"](q_k, f_0, h)
        r_k = Int.methods()["rk4"](r_k, f_0, h)
    print(f"RK1: s_{k+1} = {s_k}")
    print(f"RK2: s_{k+1} = {p_k}")
    print(f"RK3: s_{k+1} = {q_k}")
    print(f"RK4: s_{k+1} = {r_k}")

def test_integrator(N = 100):
    h = 0.1
    r = 0.5
    f_0 = lambda t, x: - r * x
    s_0 = make_state(0, 2) 
    Int = Integrator().setup(f_0, h, "rk4", n_steps=N)
    s_N = Int.f_int()(s_0)
    print(f"RK4: s_{N} = {s_N}")

def test_sin(N = 100):
    h = 0.1
    omega = 2 * jnp.pi / 7
    f_0 = lambda t, x: jnp.cos(t*omega)
    s_0 = make_state(0, 0)
    Int = Integrator(f_0, h, "rk4", n_steps=N)
    s_est = Int.integrate(s_0)
    s_true = s_est.t, jnp.sin(s_est.t*omega)/omega
    print(f"s_rk4: s_{N} = {s_est}")
    print(f"s_true: s_{N} = {s_true}")