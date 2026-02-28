import xarray as xr
from zonjets.atm2 import Atmosphere, fft2, ifft2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# atm = Atmosphere(name='data')
 
# u_tilde = fft2(atm.u) * atm.dxdy
# dudy_tilde = atm.iky * u_tilde
# dudy = jnp.real(ifft2(dudy_tilde)) * atm.dkdl

# plt.figure(figsize=(6,5))
# plt.contourf(np.array(atm.X), np.array(atm.Y), np.array(dudy), levels=60, cmap='RdBu_r')
# plt.colorbar()
# plt.title("Vorticity")
# plt.show()
 
# for k in range(100):
#     atm.calc(1000)
#     print(atm.R_beta)
#     atm.plot_zeta(show=False, save=True)
#     atm.plot_U(show=False, save=True)
#     atm.plot_Ux(show=False, save=True)
#     atm.plot_Uy(show=False, save=True)
#     # atm.plot_Ek(show=False, save=True)
#     # atm.plot_Zk(show=False, save=True)
#     # if atm.R_beta > 2:
#     #     break

