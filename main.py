from qg_atm import *
from test import test_all
import numpy as np

import matplotlib.pyplot as plt

s, f0, beta = 1, 0, 0
p, nu, r = 1, 0, 0
kappa = 0
epsilon = 0
param = s, f0, beta, p, nu, r, kappa, epsilon, 0, 0, 0
N, M = 128, 100
grid = (N, M)
method = "rk4"
initial = "dipole"
forcing = "zero"
desrciprion = param, grid, method, initial, forcing

atm = Atmosphere().setup(*desrciprion)
f_T = Integrator().setup(*atm.model(), N=M)
s_0 = atm.start()
_, q_hat = s_0

tau = 10

s_k = s_0

for k in range(tau):
    s_k = f_T(s_k)
    print(f"t = {k+1}")

_, q_hat_k = s_k
q = atm._ifft_phys(q_hat_k)
ux_hat, uy_hat = atm._u_hat(q_hat_k)
e_hat = np.abs(ux_hat)**2 + np.abs(uy_hat)**2 
z_hat = np.abs(q_hat)**2 
ux, uy = atm._ifft_phys(ux_hat), atm._ifft_phys(uy_hat)
u = np.sqrt(ux**2 + uy**2)

# plt.figure(figsize=(6,5))
# plt.contourf(atm.X, atm.Y, q, levels=50, cmap='coolwarm')
# plt.colorbar()
# plt.title("Vorticity")

plt.figure(figsize=(6,5))
plt.contourf(atm.X, atm.Y, u, levels=50, cmap='coolwarm')
plt.colorbar()
plt.title("Velocity")

# n_p = + (atm.N+1)//2
# n_m = - (atm.N)//2 

# plt.contourf(atm.Kx[:n_p, :n_p], atm.Ky[:n_p, :n_p], e_hat[:n_p, :n_p], levels=50, cmap='coolwarm')
# plt.colorbar()
# plt.title("Vorticity")

plt.show()
