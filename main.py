from qg_atm import *
from test import test_all

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def setup_model(Int, atm, io, name='base', update=None, tau=1):
    model = io.load_model(name=name)
    if update is not None:
        model = update(model)
        io.dump_model(model, name=name)
    
    params = make_params(**to_dict(model.params))
    scale = make_scale(**to_dict(model.scale))
    atm.setup(params, scale, model.method, model.initial, model.forcing)
    Int.setup(*atm.model(), n_steps=scale.M*tau)

    return Int, atm, io, model

def update(model):
    model.params.nu = 1e-5
    model.scale.N, model.scale.M = 128, 10
    model.initial = "random"
    return model

Int, atm, io, model = setup_model(Integrator(), Atmosphere(), IOManager(), name='test', update=update)
s_0 = make_state(*atm.start())
f_int = Int.f_int()
# fig, ax = plt.subplots()

T = 200

# def analitics(s):
#     ps = atm.calc(s).p
#     u = np.sqrt(ps.ux**2 + ps.uy**2)
#     plt.clf()
#     plt.contourf(ps.X, ps.Y, q, levels=50, vmin=-1, vmax=1, cmap='RdBu_r')
#     plt.colorbar()
#     plt.savefig("q.png")
#     plt.close()

def save(s, name=None):
    io.dump_state(s, model, name=name)
    io.dump_atm_state(atm.calc(s), model, all_steps=True, name=name) 

def graph(s, name=None):
    io.print_atm_state(atm.calc(s), "q", levels=30, range_val=(-1, 1), all_steps=False, colorbar=False, title=" | Vorticity", cmap='balance', name=name)

name = "u-rnd"

# s_data, model = io.load_state(name=name)
# s_k = make_state(jnp.array(s_data.t), jnp.array(s_data.y))
s_k = s_0

# save(s_k, name=name)
# for k in tqdm(range(T)):
#     s_k = f_int(s_k)
#     save(s_k, name=name)
#     graph(s_k, name=name)

ds = io.make_atm_dataset(name=name)
ds, model = io.load_atm_dataset(name=name)
q = ds['q']
u = np.sqrt(ds['ux']**2 + ds['ux']**2)
ds['u'] = np.sqrt(ds['ux']**2 + ds['ux']**2)
io.print_atm_dataset(ds, 'u', colorbar=False, title=" | Velocity", cmap='speed', name=name, fps=20, range_val="end")
io.print_atm_dataset(ds, 'q', colorbar=False, title=" | Vorticity", cmap='balance', name=name, fps=20, range_val="end")