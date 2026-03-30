import h5py, json
import tomli_w, tomllib, os
import jax.numpy as jnp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, asdict
from tqdm import tqdm

DESCRIPT = {
    'meta': {
        'name': 'qg-model',
        'coord': 'cartesian',
        'approach': 'spectral-1',
    },
    'scale': {
        'N': 10,
        'M': 10,
    },
    'params': {
        'coriolis': {
            's': 1,
            'f0': 1,
            'beta': 0
        },
        'model': {
            'p': 1,
            'nu': 0,
            'r': 0,
            'epsilon': 0,
            'kappa': 0,
        },
        'seeds': {
            'initial_seed': 42,
            'forcing_seed': 43,
        },
        'spectrum': {
            'kf': 0, 
            'dkf': 0, 
            'sigma': 0,
        },
    },
    'method': "rk4",
    'initial': "random",
    'forcing':"zero",
}

CMOCEAN = {
    'thermal': cmo.thermal,
    'haline': cmo.haline,
    'solar': cmo.solar,
    'ice': cmo.ice,
    'gray': cmo.gray,
    'oxy': cmo.oxy,
    'deep': cmo.deep,
    'dense': cmo.dense,
    'algae': cmo.algae,
    'matter': cmo.matter,
    'turbid': cmo.turbid,
    'speed': cmo.speed,
    'amp': cmo.amp,
    'tempo': cmo.tempo,
    'rain': cmo.rain,
    'phase': cmo.phase,
    'topo': cmo.topo,
    'balance': cmo.balance,
    'delta': cmo.delta,
    'curl': cmo.curl,
    'diff': cmo.diff,
    'tarn': cmo.tarn,
}

def check_cmap(cmap):
    if cmap in CMOCEAN:
        return CMOCEAN[cmap]
    else:
        return cmap

@dataclass
class MetaConfig:
    name: str
    coord: str
    approach: str

@dataclass
class ParamsConfig:
    s : int
    f0: float
    beta: float
    p: int
    nu: float
    r: float
    epsilon: float
    kappa: float
    initial_seed: int
    forcing_seed: int
    kf: float
    dkf: float
    sigma: float

@dataclass
class ScaleConfig:
    N: int
    M: int

@dataclass
class ModelConfig:
    meta: MetaConfig
    scale: ScaleConfig
    params: ParamsConfig
    method: str
    initial: str
    forcing: str

@dataclass
class StateConfig:
    t: jnp.ndarray
    y: jnp.ndarray

@dataclass
class AtmosphereStateConfig:
    alpha: np.ndarray
    t: np.ndarray
    q: np.ndarray
    psi: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    r: np.ndarray
    k: np.ndarray
    
def to_dict(config):
    return asdict(config)

def make_model():
    model = ModelConfig(
            meta = MetaConfig(**DESCRIPT['meta']),
            scale = ScaleConfig(**DESCRIPT['scale']),
            params = ParamsConfig(
                **DESCRIPT['params']['coriolis'], 
                **DESCRIPT['params']['model'], 
                **DESCRIPT['params']['seeds'],
                **DESCRIPT['params']['spectrum'],
            ),
            method=DESCRIPT['method'],
            initial=DESCRIPT['initial'],
            forcing=DESCRIPT['forcing'],
    )
    return model


class IOManager:
    def __init__(self, data_path=None, dir_name="data"):
        if data_path is None or data_path == "":
            self.data_path = dir_name
        else: 
            self.data_path = f"{data_path}/{dir_name}"

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def load_model(self, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        if not os.path.exists(calc_path):
            os.mkdir(calc_path)
        
        filename = "descript"

        if not os.path.exists(f"{calc_path}/{filename}.toml"):
            with open(f"{calc_path}/{filename}.toml", "wb") as f:
                tomli_w.dump(DESCRIPT, f)

        with open(f"{calc_path}/{filename}.toml", "rb") as f:
            descript = tomllib.load(f)

        model = ModelConfig(
            meta = MetaConfig(**descript['meta']),
            scale = ScaleConfig(**descript['scale']),
            params = ParamsConfig(
                **descript['params']['coriolis'], 
                **descript['params']['model'], 
                **descript['params']['seeds'],
                **descript['params']['spectrum'],
            ),
            method=descript['method'],
            initial=descript['initial'],
            forcing=descript['forcing'],
        )

        return model

    def dump_model(self, model, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        if not os.path.exists(calc_path):
            os.mkdir(calc_path)
        
        filename = "descript"

        descript = {
            'meta': asdict(model.meta),
            'scale': asdict(model.scale),
            'params': {
                    'coriolis': {
                        's': model.params.s,
                        'f0': model.params.f0,
                        'beta': model.params.beta,
                    },
                    'model': {
                        'p': model.params.p,
                        'nu': model.params.nu,
                        'r': model.params.r,
                        'epsilon': model.params.epsilon,
                        'kappa': model.params.kappa,
                    },
                    'seeds': {
                        'initial_seed': model.params.initial_seed,
                        'forcing_seed': model.params.forcing_seed,
                    },
                    'spectrum': {
                        'kf': model.params.kf, 
                        'dkf': model.params.dkf, 
                        'sigma': model.params.sigma,
                    },
            },
            'method': model.method,
            'initial': model.initial,
            'forcing': model.forcing,
        }

        with open(f"{calc_path}/{filename}.toml", "wb") as f:
            tomli_w.dump(descript, f)

    def dump_state(self, state, model, all_steps=False, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        descript = to_dict(model)

        calc_path = f"{self.data_path}/{calc_name}"
        if not os.path.exists(calc_path):
            os.mkdir(calc_path)

        state_path = f"{calc_path}/states"
        if not os.path.exists(state_path):
            os.mkdir(state_path)

        state_name_last = "state"
        state_name_step = f"state_{round(state.t)}"

        if all_steps:
            with h5py.File(f"{state_path}/{state_name_step}.h5", "w") as f:
                f.create_dataset("y", data=state.y)
                f.attrs["t"] = state.t
                f.attrs["descript"] = json.dumps(descript)
        
        with h5py.File(f"{state_path}/{state_name_last}.h5", "w") as f:
            f.create_dataset("y", data=state.y)
            f.attrs["t"] = state.t
            f.attrs["descript"] = json.dumps(descript)

    def load_state(self, num=None, name=None, parse_model=False):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        state_path = f"{calc_path}/states"

        if num is None:
            state_name = "state"
        else:
            state_name = f"state_{num}"

        with h5py.File(f"{state_path}/{state_name}.h5", "r") as f:
            y = jnp.array(f["y"][:])
            t = jnp.array(float(f.attrs["t"]))
            descript = json.loads(f.attrs["descript"])
        
        state = StateConfig(t, y)

        if parse_model:
            model = ModelConfig(
                meta = MetaConfig(**descript['meta']),
                scale = ScaleConfig(**descript['scale']),
                params = ParamsConfig(**descript['params']),
                method=descript['method'],
                initial=descript['initial'],
                forcing=descript['forcing'],
            )
            return state, model

        else:
            return state
            
    def dump_atm_state(self, atm_state, model, all_steps=False, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        descript = to_dict(model)

        calc_path = f"{self.data_path}/{calc_name}"
        if not os.path.exists(calc_path):
            os.mkdir(calc_path)

        atm_state_path = f"{calc_path}/atm-states"
        if not os.path.exists(atm_state_path):
            os.mkdir(atm_state_path)

        atm_state_name_last = "atm-state"
        atm_state_name_step = f"atm-state_{int(np.round(atm_state.t))}"
        
        if all_steps:
            with h5py.File(f"{atm_state_path}/{atm_state_name_step}.h5", "w") as f:
                f.attrs["t"] = atm_state.t
                f.attrs["alpha"] = atm_state.alpha
                f.attrs["descript"] = json.dumps(descript)
                f.create_dataset("q", data=atm_state.q)
                f.create_dataset("psi", data=atm_state.psi)
                f.create_dataset("ux", data=atm_state.ux)
                f.create_dataset("uy", data=atm_state.uy)
                f.create_dataset("r", data=atm_state.r)
                f.create_dataset("k", data=atm_state.k)

        with h5py.File(f"{atm_state_path}/{atm_state_name_last}.h5", "w") as f:
            f.attrs["t"] = atm_state.t
            f.attrs["alpha"] = atm_state.alpha
            f.attrs["descript"] = json.dumps(descript)
            f.create_dataset("q", data=atm_state.q)
            f.create_dataset("psi", data=atm_state.psi)
            f.create_dataset("ux", data=atm_state.ux)
            f.create_dataset("uy", data=atm_state.uy)
            f.create_dataset("r", data=atm_state.r)
            f.create_dataset("k", data=atm_state.k)


    def load_atm_state(self, num=None, name=None, parse_model=False):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        atm_state_path = f"{calc_path}/atm-states"

        if num is None:
            atm_state_name = "atm-state"
        else:
            atm_state_name = f"atm-state_{num}"

        with h5py.File(f"{atm_state_path}/{atm_state_name}.h5", "r") as f:
            t = np.array(float(f.attrs["t"]))
            alpha = np.array(float(f.attrs["alpha"]))
            descript = json.loads(f.attrs["descript"])
            q = np.array(f["q"][:])
            psi = np.array(f["psi"][:])
            ux = np.array(f["ux"][:])
            uy = np.array(f["uy"][:])
            r = np.array(f["r"][:])
            k = np.array(f["k"][:])

        atm_state = AtmosphereStateConfig(alpha, t, q, psi, ux, uy, r, k)

        if parse_model:
            model = ModelConfig(
                meta = MetaConfig(**descript['meta']),
                scale = ScaleConfig(**descript['scale']),
                params = ParamsConfig(**descript['params']),
                method=descript['method'],
                initial=descript['initial'],
                forcing=descript['forcing'],
            )
            return atm_state, model

        else:
            return atm_state

    def state_to_dataset(self, atm_state, model):
        descript = to_dict(model)
        dataset_descript = {}

        for key, value in descript.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    dataset_descript[f"{key}.{k}"] = v
            else:
                dataset_descript[key] = value

        kx, ky = np.meshgrid(atm_state.k, atm_state.k, indexing='ij')
        k1 = np.sqrt(kx**2 + ky**2)

        ds_state = xr.Dataset(
            data_vars={
                "q":   (("x", "y"), atm_state.q),
                "psi": (("x", "y"), atm_state.psi),
                "ux":  (("x", "y"), atm_state.ux),
                "uy":  (("x", "y"), atm_state.uy),
            },

            coords={
                "x": atm_state.r,
                "y": atm_state.r,
                "k": atm_state.k,
                "k_abs": np.arange((np.floor(k1).astype(int)).max() + 1),
            },

            attrs={
                **dataset_descript,
                "alpha": float(atm_state.alpha),
            }
        )

        ds_state = ds_state.expand_dims(t=[atm_state.t])

        return ds_state

    def make_atm_dataset(self, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        dataset_path = f"{calc_path}/atm.nc"

        datasets = []
        atm_last = self.load_atm_state(name=name)
        T = int(np.round(atm_last.t))

        print("\nPreparing dataset...")
        for k in tqdm(range(T+1)):
            atm_state, model = self.load_atm_state(num=k, name=name, parse_model=True)
            ds_state = self.state_to_dataset(atm_state, model)
            datasets.append(ds_state)
        dataset_full = xr.concat(datasets, dim="t")

        return dataset_full

    def dump_atm_dataset(self, dataset, ds_name=None, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"

        if ds_name is None:
            dataset_path = f"{calc_path}/atm.nc"
        else:
            datasets_path = f"{calc_path}/datasets"
            if not os.path.exists(datasets_path):
                os.mkdir(datasets_path)
            dataset_path = f"{datasets_path}/atm-{ds_name}.nc"

        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in dataset.data_vars
        }

        dataset.to_netcdf(dataset_path, engine="netcdf4", encoding=encoding)


    def load_atm_dataset(self, ds_name=None, name=None, parse_model=False):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"

        if ds_name is None:
            dataset_path = f"{calc_path}/atm.nc"
        else:
            datasets_path = f"{calc_path}/datasets"
            dataset_path = f"{datasets_path}/atm-{ds_name}.nc"
        
        print("\nLoading dataset...")
        ds = xr.open_dataset(dataset_path).load()
        descript_dataset = ds.attrs
        
        if parse_model:
            descript = {}
            for key, value in descript_dataset.items():
                if len(key.split("."))==2:
                    key_maj, key_min = key.split(".")
                    if not (key_maj in descript):
                        descript[key_maj] = {}
                    if isinstance(value, np.int64):
                        descript[key_maj][key_min] = int(value)
                    elif isinstance(value, np.float64):
                        descript[key_maj][key_min] = float(value)
                    else:
                        descript[key_maj][key_min] = value
                else:
                    descript[key] = value

            model = ModelConfig(
                meta = MetaConfig(**descript['meta']),
                scale = ScaleConfig(**descript['scale']),
                params = ParamsConfig(**descript['params']),
                method=descript['method'],
                initial=descript['initial'],
                forcing=descript['forcing'],
            )
            return ds, model
        else:
            return ds

    def print_state_field(self, ds_state, key, cmap='balance', levels=50, colorbar=True, range_val = "all", title="", all_steps=False, show=False, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        graph_path = f"{calc_path}/graphs"
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

        graph_name_step = f"atm-{key}_{int(np.round(ds_state['t']))}.png"
        graph_name_last = f"atm-{key}.png"

        z = ds_state[key].isel(t=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        if isinstance(range_val, tuple):
            vmin, vmax = range_val
        else:
            vmin = float(np.min(z)) 
            vmax = float(np.max(z))

        if colorbar:
            im = ax.contourf(ds_state["x"], ds_state["y"], z.T, vmin=vmin, vmax=vmax, cmap=check_cmap(cmap), levels=levels)
            fig.colorbar(im, ax=ax)
                        
        ax.contourf(ds_state["x"], ds_state["y"], z.T, vmin=vmin, vmax=vmax, cmap=check_cmap(cmap), levels=levels)
        base_title = f"t = {int(np.round(ds_state['t']))}"
        full_title = base_title + title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(full_title)
        
        if all_steps:
            plt.savefig(f"{graph_path}/{graph_name_step}")

        plt.savefig(f"{graph_path}/{graph_name_last}")

        if show:
            plt.show()

        plt.close()

    def print_state_quantity(self, ds_state, key, style='seaborn-v0_8', color='blue', range_val = (None, None), title="", log=None, all_steps=False, show=False, name=None):
        plt.style.use(style)
        
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"
        calc_path = f"{self.data_path}/{calc_name}"
        graph_path = f"{calc_path}/graphs"
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

        graph_name_step = f"atm-{key}_{int(np.round(ds_state['t']))}.png"
        graph_name_last = f"atm-{key}.png"

        z = ds_state[key].isel(t=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        if isinstance(range_val, tuple):
            vmin, vmax = range_val
            if not((vmin is None) and (vmax is None)):
                if vmin is None:
                    vmin = float(np.min(z)) 
                if vmax is None:
                    vmax = float(np.max(z))
        else:
            vmin = float(z.min()) 
            vmax = float(z.max())
        
        if log is None:
            ax.plot(z, color=color)
            ax.set_ylim(vmin, vmax)
        elif log == "log":
            ax.semilogy(z, color=color)
            if not(isinstance(range_val, tuple) and range_val[0] is None and range_val[1] is None):
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)
        elif log == "loglog":
            ax.loglog(z, color=color)
            if not(isinstance(range_val, tuple) and range_val[0] is None and range_val[1] is None):
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)
            
        base_title = f"t = {int(np.round(ds_state['t']))}"
        full_title = base_title + title
        ax.set_xlabel(z.dims[0])
        ax.set_ylabel(key)
        ax.set_title(full_title)
        
        if all_steps:
            plt.savefig(f"{graph_path}/{graph_name_step}")

        plt.savefig(f"{graph_path}/{graph_name_last}")
        
        if show:
            plt.show()
            
        plt.close()
        

    def print_dataset_field(self, ds, key, cmap='balance', levels=50, colorbar=True, range_val = "all", fps=30, title="", name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        
        graph_name = f"atm-{key}.mp4"

        fig, ax = plt.subplots(figsize=(5, 5))
        z = ds[key]

        if range_val == "start":
            vmin = float(z.isel(t=0).min()) 
            vmax = float(z.isel(t=0).max())
        elif range_val == "end":
            vmin = float(z.isel(t=len(ds["t"])-1).min()) 
            vmax = float(z.isel(t=len(ds["t"])-1).max())
        elif isinstance(range_val, tuple):
            vmin, vmax = range_val
        else:
            vmin = float(z.min()) 
            vmax = float(z.max())
        
        if colorbar:
            im = ax.contourf(
                ds["x"],
                ds["y"],
                z.isel(t=0).T,
                vmin=vmin,
                vmax=vmax,
                cmap=check_cmap(cmap),
                levels=levels,
            )
            fig.colorbar(im, ax=ax)

        def animate(i):
            ax.clear()
            ax.contourf(
                ds["x"],
                ds["y"],
                z.isel(t=i).T,
                vmin=vmin,
                vmax=vmax,
                cmap=check_cmap(cmap),
                levels=levels,
            )
            base_title = f"t = {int(np.round(ds['t'].values[i]))}"
            full_title = base_title + title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(full_title)

        ani = FuncAnimation(fig, animate, frames=len(ds["t"]))
        print(f"Saving animation of {key}...")
        ani.save(f"{calc_path}/{graph_name}", fps=fps)

    def print_dataset_quantity(self, ds, key, style='seaborn-v0_8', color='blue', range_val = (None, None), title="", log=None, fps=30, name=None):
        plt.style.use(style)
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        
        graph_name = f"atm-{key}.mp4"

        fig, ax = plt.subplots(figsize=(5, 5))
        z = ds[key]

        if isinstance(range_val, tuple):
            vmin, vmax = range_val
            if not((vmin is None) and (vmax is None)):
                if vmin is None:
                    vmin = float(np.min(z)) 
                if vmax is None:
                    vmax = float(np.max(z))
        elif range_val == "start":
            vmin = float(z.isel(t=0).min()) 
            vmax = float(z.isel(t=0).max())
        elif range_val == "end":
            vmin = float(z.isel(t=len(ds["t"])-1).min()) 
            vmax = float(z.isel(t=len(ds["t"])-1).max())
        else:
            vmin = float(z.min()) 
            vmax = float(z.max())
            
        def animate(i):
            ax.clear()
            if log is None:
                ax.plot(z.isel(t=i), color=color)
                ax.set_ylim(vmin, vmax)
            elif log == "log":
                ax.semilogy(z.isel(t=i), color=color)
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)
            elif log == "loglog":
                ax.loglog(z.isel(t=i), color=color)
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)
            base_title = f"t = {int(np.round(ds['t'].values[i]))}"
            full_title = base_title + title
            ax.set_xlabel(z.isel(t=0).dims[0])
            ax.set_ylabel(key)
            ax.set_title(full_title)

        ani = FuncAnimation(fig, animate, frames=len(ds["t"]))
        print(f"Saving animation of {key}...")
        ani.save(f"{calc_path}/{graph_name}", fps=fps)

    def print_dataset_statistics(self, ds, key, style='seaborn-v0_8', color='blue', range_val = (None, None), title="", log=None, name=None):
        plt.style.use(style)
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        
        graph_name = f"atm-{key}.png"

        fig, ax = plt.subplots(figsize=(5, 5))
        z = ds[key]

        if isinstance(range_val, tuple):
            vmin, vmax = range_val
            if not((vmin is None) and (vmax is None)):
                if vmin is None:
                    vmin = float(np.min(z)) 
                if vmax is None:
                    vmax = float(np.max(z))
        else:
            vmin = float(z.min()) 
            vmax = float(z.max())
            
        if log is None:
            ax.plot(z, color=color)
            if not(isinstance(range_val, tuple) and range_val[0] is None and range_val[1] is None):
                dvmin, dvmax = 0, (vmax-vmin)/5
                ax.set_ylim(vmin, vmax+dvmax)
        elif log == "log":
            ax.semilogy(z, color=color)

            if not(isinstance(range_val, tuple) and range_val[0] is None and range_val[1] is None):
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)
        elif log == "loglog":
            ax.loglog(z, color=color)
            if not(isinstance(range_val, tuple) and range_val[0] is None and range_val[1] is None):
                dvmin, dvmax = 0, 10**(np.log10(vmax) + 0.5)
                if vmin<=0:
                    dvmin = vmin+10**(np.log10(vmax) - 5)
                ax.set_ylim(vmin + dvmin, vmax + dvmax)

        ax.set_xlabel(z.dims[0])
        ax.set_ylabel(key)
        ax.set_title(title)

        plt.savefig(f"{calc_path}/{graph_name}")
        plt.close()