import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .const import *
from .timer import Timer
from .integrator import Integrator, make_state
from .atmosphere import Atmosphere, make_scale, make_grid, make_params, make_atmosphere_state
from .io_manager import IOManager, to_dict, make_model
from .diagnostics import Diagnostics

NAME = 'report'


class Config:
    def __init__(self, experiment=None, descript = None, n_iter= None, t_iter=None, print_scan=True, print_state=None, print_dataset=None, ps_args=None, pds_args=None, first_state=None):
        self.experiment = experiment
        self.descript = descript
        self.n_iter = n_iter
        self.t_iter = t_iter
        self.model = make_model()
        self.print_scan = print_scan
        self.print_state = print_state
        self.print_dataset = print_dataset
        self.ps_args = ps_args
        self.pds_args = pds_args
        self.first_state = first_state

        if isinstance(descript, dict):
            for key, value in descript.items():
                if hasattr(self.model.meta, key):
                    setattr(self.model.meta, key, value)
                if hasattr(self.model.scale, key):
                    setattr(self.model.scale, key, value)
                if hasattr(self.model.params, key):
                    setattr(self.model.params, key, value)
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
            
class Driver:
    def __init__(self, config=None):
        self.Int = Integrator()
        self.atm = Atmosphere()
        self.io = IOManager()
        self.diag = Diagnostics()
        self.timer = Timer()
        self.setup(config)

    def setup(self, config):
        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.name = self.config.experiment
        self.model = self.config.model
        if self.config.t_iter is None:
            self.t_iter = 1
        else: 
            self.t_iter = self.config.t_iter

        model = self.model
        params = make_params(**to_dict(model.params))
        scale = make_scale(**to_dict(model.scale))
        self.atm.setup(params, scale, model.method, model.initial, model.forcing)
        self.Int.setup(*self.atm.model(), n_steps=scale.M*self.t_iter)
        self.io.dump_model(model, name=self.name)
        self.state = make_state(*self.atm.start())
        self.f_int = self.Int.f_int()
        self.devices = self.Int.devices()
        
        return self

    def run(self, n_iter=1):
        if self.config.n_iter is not None:
            n_iter = self.config.n_iter
        state = self.dataload()
        state = self.preprocess(state, n_iter)
        for k in range(n_iter):
            state = self.pipeline(state, k, n_iter)
        return self.postprocess(state)

    def dataload(self):
        if isinstance(self.config.first_state, int):
            state_data = self.io.load_state(self.config.first_state, name=self.name)
            state = make_state(**to_dict(state_data))
        elif self.config.first_state == "last":
            state_data = self.io.load_state(name=self.name)
            state = make_state(**to_dict(state_data))
        else:
            state = self.state
        return state

    def preprocess(self, state, n_iter):
        self.timer.check()
        self.save(state)
        self.scan(state, 0, n_iter)
        if self.config.print_state:
            self.print_state(state)
        return state

    def pipeline(self, state, k_iter, n_iter):
        new_state = self.f_int(state)
        self.save(new_state)
        self.scan(new_state, k_iter+1, n_iter)
        if self.config.print_state is not None:
            self.print_state(state)
        return new_state

    def postprocess(self, state=None):
        dataset = self.io.make_atm_dataset(name=self.name, step=self.t_iter)
        dataset = self.calc(dataset)
        self.io.dump_atm_dataset(dataset, name=self.name)
        if self.config.print_dataset is not None:
            self.print_dataset(dataset)
        return dataset

    def scan(self, s, k_iter, n_iter):
        if self.config.print_scan is False:
            return None
        ds_state = self.io.state_to_dataset(self.atm.calc(s), self.model)
        KE = float(self.diag.mean_energy(ds_state)[0])
        VE = float(self.diag.mean_enstrophy(ds_state)[0])
        time_iter = self.timer.check()
        if k_iter == 0:
            self.speed = 0
        else:
            self.speed = (self.speed * (k_iter-1) + time_iter)/(k_iter)

        logging = f"KE: {KE:.2e}, VE: {VE:.2e} | time-iter = {self.timer.pretty_time(time_iter)} | {k_iter:>{int(np.log10(n_iter))+1}}/{n_iter} [{self.timer.standart_time(self.speed*k_iter)}<{self.timer.standart_time(time_iter*(n_iter-k_iter))}]"
        print(f"\r{logging}", end="")

    def save(self, s):
        self.io.dump_state(s, self.model, name=self.name)
        self.io.dump_atm_state(self.atm.calc(s), self.model, all_steps=True, name=self.name) 

    def calc(self, ds):
        ds['u'] = self.diag.velocity(ds)
        ds['e_mean'] = self.diag.mean_energy(ds)
        ds['z_mean'] = self.diag.mean_enstrophy(ds)
        ds['e_spec'] = self.diag.energy_spectrum(ds)
        ds['z_spec'] = self.diag.enstrophy_spectrum(ds)
        ds['e_flow'] = self.diag.energy_flow(ds)
        ds['z_flow'] = self.diag.enstrophy_flow(ds)
        return ds

    def print_state(self, state):
        cfg = {
            'colorbar': False,
            'show': False,
            'q_cmap': 'curl',
            'e_color': 'blue',
            'z_color': 'green',
        }
        if isinstance(self.config.ps_args, dict):
            for key, value in self.config.ps_args.items():
                if key in cfg:
                    cfg[key] = value
        
        if self.config.print_state == "min":
            ds_state = self.io.state_to_dataset(self.atm.calc(state), self.model)
            self.io.print_state_field(ds_state, 'q', show=cfg['show'], colorbar=cfg['colorbar'], title=" | Vorticity", cmap=cfg['q_cmap'], range_val="all", name=self.name)
        elif self.config.print_state == "std":
            ds_state = self.io.state_to_dataset(self.atm.calc(state), self.model)
            ds_state['u'] = self.diag.velocity(ds_state)
            ds_state['e_spec'] = self.diag.energy_spectrum(ds_state)
            ds_state['z_spec'] = self.diag.enstrophy_spectrum(ds_state)
            self.io.print_state_field(ds_state, 'q', show=cfg['show'], colorbar=cfg['colorbar'], title=" | Vorticity", cmap=cfg['q_cmap'], range_val="all", name=self.name)
            self.io.print_state_field(ds_state, 'u', show=cfg['show'], colorbar=cfg['colorbar'], title=" | Velocity", cmap='speed', range_val="all", name=self.name)
            self.io.print_state_quantity(ds_state, "e_spec", show=cfg['show'], color=cfg['e_color'], title=" | Energy Spectrum", log='loglog', name=self.name)
            self.io.print_state_quantity(ds_state, "z_spec", show=cfg['show'], color=cfg['z_color'], title=" | Enstrophy Spectrum", log='loglog', name=self.name)

    def print_dataset(self, dataset):
        cfg = {
            'q_range': "all",
            'u_range': "all",
            'fps': 20,
            'q_cmap': 'curl',
            'colorbar': False,
            'show': False,
            'e_color': 'blue',
            'z_color': 'green',

        }
        if isinstance(self.config.pds_args, dict):
            for key, value in self.config.pds_args.items():
                if key in cfg:
                    cfg[key] = value

        if self.config.print_dataset == "min":
            self.io.print_dataset_field(dataset, 'q', colorbar=cfg['colorbar'], title=" | Vorticity", cmap=cfg['q_cmap'], fps=cfg['fps'], range_val=cfg['q_range'], name=self.name)
            self.io.print_dataset_field(dataset, 'u', colorbar=cfg['colorbar'], title=" | Velocity", cmap='speed', fps=cfg['fps'], range_val=cfg['u_range'], name=self.name)
        elif self.config.print_dataset == "std":
            self.io.print_dataset_field(dataset, 'q', colorbar=cfg['colorbar'], title=" | Vorticity", cmap=cfg['q_cmap'], fps=cfg['fps'], range_val=cfg['q_range'], name=self.name)
            self.io.print_dataset_field(dataset, 'u', colorbar=cfg['colorbar'], title=" | Velocity", cmap='speed', fps=cfg['fps'], range_val=cfg['u_range'], name=self.name)
            self.io.print_dataset_quantity(dataset, "e_spec", color=cfg['e_color'], title=" | Energy Spectrum", log='loglog', fps=cfg['fps'], range_val="all", name=self.name)
            self.io.print_dataset_quantity(dataset, "z_spec", color=cfg['z_color'], title=" | Enstrophy Spectrum", log='loglog', fps=cfg['fps'], range_val="all", name=self.name)
        elif self.config.print_dataset == "all":
            self.io.print_dataset_field(dataset, 'q', colorbar=cfg['colorbar'], title=" | Vorticity", cmap=cfg['q_cmap'], fps=cfg['fps'], range_val=cfg['q_range'], name=self.name)
            self.io.print_dataset_field(dataset, 'u', colorbar=cfg['colorbar'], title=" | Velocity", cmap='speed', fps=cfg['fps'], range_val=cfg['u_range'], name=self.name)
            self.io.print_dataset_quantity(dataset, "e_spec", color=cfg['e_color'], title=" | Energy Spectrum", log='loglog', fps=cfg['fps'], range_val="all", name=self.name)
            self.io.print_dataset_quantity(dataset, "z_spec", color=cfg['z_color'], title=" | Enstrophy Spectrum", log='loglog', fps=cfg['fps'], range_val="all", name=self.name)
            self.io.print_dataset_statistics(dataset,'e_mean', color=cfg['e_color'], range_val=(0, None), title="Mean Energy", name=self.name, log="log")
            self.io.print_dataset_statistics(dataset,'z_mean', color=cfg['z_color'], range_val=(0, None), title="Mean Enstrophy", name=self.name, log="log")
            self.io.print_dataset_quantity(dataset, "e_flow", color=cfg['e_color'], title=" | Energy Flow", fps=cfg['fps'], range_val="all", name=self.name)
            self.io.print_dataset_quantity(dataset, "z_flow", color=cfg['z_color'], title=" | Enstrophy Flow", fps=cfg['fps'], range_val="all", name=self.name)
