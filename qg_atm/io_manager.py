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
class PhysicalStateConfig:
    q: np.ndarray
    psi: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    X: np.ndarray
    Y: np.ndarray

@dataclass
class SpectralStateConfig:
    q_hat: np.ndarray
    psi_hat: np.ndarray
    ux_hat: np.ndarray
    uy_hat: np.ndarray
    Kx: np.ndarray
    Ky: np.ndarray

@dataclass
class AtmosphereStateConfig:
    alpha: np.ndarray
    t: np.ndarray
    p: PhysicalStateConfig
    s: SpectralStateConfig
    
def to_dict(config):
    return asdict(config)

class IOManager:
    def __init__(self, data_path=None, dir_name="data"):
        if data_path is None or data_path == "":
            self.data_path = dir_name
        else: 
            self.data_path = f"{data_path}/{dir_name}"

        self.config_path = f"{self.data_path}/config"

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        
        if not os.path.exists(self.config_path):
            os.mkdir(self.config_path)

        with open(f"{self.config_path}/descript.toml", "wb") as f:
            tomli_w.dump(DESCRIPT, f)

    def load_model(self, name=None):
        if name is None:
            filename = "descript"
        else:
            filename = f"descript-{name}"

        if not os.path.exists(f"{self.config_path}/{filename}.toml"):
            with open(f"{self.config_path}/{filename}.toml", "wb") as f:
                tomli_w.dump(DESCRIPT, f)

        with open(f"{self.config_path}/{filename}.toml", "rb") as f:
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
            filename = "descript"
        else:
            filename = f"descript-{name}"

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

        with open(f"{self.config_path}/{filename}.toml", "wb") as f:
            tomli_w.dump(descript, f)

    def dump_state(self, state, model, all_steps=False, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

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

    def load_state(self, num=None, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

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

        model = ModelConfig(
            meta = MetaConfig(**descript['meta']),
            scale = ScaleConfig(**descript['scale']),
            params = ParamsConfig(**descript['params']),
            method=descript['method'],
            initial=descript['initial'],
            forcing=descript['forcing'],
        )

        return state, model

    def dump_atm_state(self, atm_state, model, all_steps=False, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

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

                g_p = f.create_group("physical")
                for name, value in atm_state.p._asdict().items():
                    g_p.create_dataset(name, data=value)

                g_s = f.create_group("spectral")
                for name, value in atm_state.s._asdict().items():
                    g_s.create_dataset(name, data=value)
        
        with h5py.File(f"{atm_state_path}/{atm_state_name_last}.h5", "w") as f:
            f.attrs["t"] = atm_state.t
            f.attrs["alpha"] = atm_state.alpha
            f.attrs["descript"] = json.dumps(descript)

            g_p = f.create_group("physical")
            for name, value in atm_state.p._asdict().items():
                g_p.create_dataset(name, data=value)

            g_s = f.create_group("spectral")
            for name, value in atm_state.s._asdict().items():
                g_s.create_dataset(name, data=value)

    def load_atm_state(self, num=None, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

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
            physical = PhysicalStateConfig(**{
                k: f["physical"][k][:]
                for k in f["physical"]
            })

            spectral = SpectralStateConfig(**{
                k: f["spectral"][k][:]
                for k in f["spectral"]
            })

        
        atm_state = AtmosphereStateConfig(alpha, t, physical, spectral)

        model = ModelConfig(
            meta = MetaConfig(**descript['meta']),
            scale = ScaleConfig(**descript['scale']),
            params = ParamsConfig(**descript['params']),
            method=descript['method'],
            initial=descript['initial'],
            forcing=descript['forcing'],
        )

        return atm_state, model

    def state_to_dataset(self, atm_state, model):
        p, s = atm_state.p, atm_state.s
        descript = to_dict(model)
        dataset_descript = {}

        for key, value in descript.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    dataset_descript[f"{key}.{k}"] = v
            else:
                dataset_descript[key] = value

        ds = xr.Dataset(
            data_vars={
                "q":   (("x", "y"), p.q),
                "psi": (("x", "y"), p.psi),
                "ux":  (("x", "y"), p.ux),
                "uy":  (("x", "y"), p.uy),

                "q_hat":   (("kx", "ky"), s.q_hat),
                "psi_hat": (("kx", "ky"), s.psi_hat),
                "ux_hat":  (("kx", "ky"), s.ux_hat),
                "uy_hat":  (("kx", "ky"), s.uy_hat),
            },

            coords={
                "x": p.X[:, 0],
                "y": p.Y[0, :],

                "kx": s.Kx[:, 0],
                "ky": s.Ky[0, :],
            },

            attrs={
                **dataset_descript,
                "alpha": float(atm_state.alpha),
            }
        )

        return ds

    def encode_complex_dataset(self, ds):
        ds = ds.copy()
        for var in list(ds.data_vars):
            if np.iscomplexobj(ds[var]):
                ds[var + "_real"] = ds[var].real
                ds[var + "_imag"] = ds[var].imag
                ds = ds.drop_vars(var)
        return ds

    def decode_complex_dataset(self, ds):
        ds = ds.copy()
        vars_to_check = list(ds.data_vars)

        for var in vars_to_check:
            if var.endswith("_real"):
                base = var[:-5]
                imag_name = base + "_imag"

                if imag_name in ds:
                    ds[base] = ds[var] + 1j * ds[imag_name]
                    ds = ds.drop_vars([var, imag_name])
        return ds

    def make_atm_dataset(self, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        dataset_path = f"{calc_path}/atm.nc"

        datasets = []
        atm_last, model = self.load_atm_state(name=name)
        T = int(np.round(atm_last.t))

        print("Preparing dataset...")
        for k in tqdm(range(T+1)):
            atm_state, model = self.load_atm_state(num=k, name=name)
            ds = self.state_to_dataset(atm_state, model)
            ds = ds.expand_dims(t=[atm_state.t])
            datasets.append(ds)

        ds_full = xr.concat(datasets, dim="t")

        ds_real = self.encode_complex_dataset(ds_full)
        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds_real.data_vars
        }
        ds_real.to_netcdf(dataset_path, engine="netcdf4", encoding=encoding)
        print("Prepared.")
        return ds_full


    def load_atm_dataset(self, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

        calc_path = f"{self.data_path}/{calc_name}"
        dataset_path = f"{calc_path}/atm.nc"
        
        print("Loading dataset...")
        ds_real = xr.open_dataset(dataset_path).load()
        ds = self.decode_complex_dataset(ds_real)
        descript_dataset = ds.attrs
        print("Loaded.")
        
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

    def print_atm_state(self, state, key, cmap='balance', levels=50, colorbar=True, range_val = "all", title="", all_steps=True, name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"
        calc_path = f"{self.data_path}/{calc_name}"
        graph_path = f"{calc_path}/graph"
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

        graph_name_step = f"atm-{key}_{int(np.round(state.t))}.png"
        graph_name_last = f"atm-{key}.png"

        z = getattr(state.p, key)
        fig, ax = plt.subplots(figsize=(5, 5))
        if isinstance(range_val, tuple):
            vmin, vmax = range_val
        else:
            vmin = float(np.min(z)) 
            vmax = float(np.max(z))

        if colorbar:
            im = ax.contourf(state.p.X, state.p.Y, z, vmin=vmin, vmax=vmax, cmap=check_cmap(cmap), levels=levels)
            fig.colorbar(im, ax=ax)
        
        ax.contourf(state.p.X, state.p.Y, z, vmin=vmin, vmax=vmax, cmap=check_cmap(cmap), levels=levels)
        base_title = f"t = {int(np.round(state.t))}"
        full_title = base_title + title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(full_title)
        
        if all_steps:
            plt.savefig(f"{graph_path}/{graph_name_step}")

        plt.savefig(f"{graph_path}/{graph_name_last}")

        plt.close()

    def print_atm_dataset(self, ds, key, cmap='balance', levels=50, colorbar=True, range_val = "all", fps=30, title="", name=None):
        if name is None:
            calc_name = "calc"
        else:
            calc_name = f"calc-{name}"

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
                z.isel(t=0),
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
                z.isel(t=i),
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
        print("Saved.")