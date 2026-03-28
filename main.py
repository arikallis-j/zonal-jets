from qg_atm import *
from test import test_all

import numpy as np
import matplotlib.pyplot as plt

Int, atm = Integrator(), Atmosphere()
N, M = 128, 10
tau = 1
method = "rk4"
initial = "random"
forcing = "zero"
params = make_param(nu=1e-10, p=2) #nu=1e-5, 
scale = make_scale(N, M)
atm.setup(params, scale, method, initial, forcing)
Int.setup(*atm.model(), n_steps=M*tau)

X, Y = np.array(atm.grid.X), np.array(atm.grid.Y)

s_0 = make_state(*atm.start())
f_int = Int.f_int()


fig, ax = plt.subplots()

def analitics(s):
    p = atm.calc(s)
    q = np.array(p.q.block_until_ready())
    ux = np.array(p.ux.block_until_ready())
    uy = np.array(p.uy.block_until_ready())
    u = np.sqrt(ux**2 + uy**2)
    plt.clf()
    plt.contourf(X, Y, q, levels=50, vmin=-1, vmax=1, cmap='RdBu_r')
    # plt.contourf(X, Y, u, levels=50, vmin=0, vmax=0.1, cmap='RdBu_r')
    plt.colorbar()
    plt.savefig("q.png")
    plt.close()

s_k = s_0

analitics(s_k)
for k in range(1_000):
    with Timer():
        s_k = f_int(s_k)
        analitics(s_k)