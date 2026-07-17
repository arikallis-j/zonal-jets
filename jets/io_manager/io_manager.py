import tomli_w, tomli
import os, shutil
import h5py, json
import xarray as xr

class IOManager:
    def __init__(self, path=None, dir_name="data", exp_name=None):
        if path is None or path == "":
            path = "."
        self.path = f"{path}/{dir_name}"
        os.makedirs(self.path, exist_ok=True)

        if exp_name is not None:
            self.setup(exp_name)

    def setup(self, exp_name="test"):
        self.exp_name = exp_name
        self.exp_path = f"{self.path}/{exp_name}"
        self.state_path = f"{self.exp_path}/states"
        self.dataset_path = f"{self.exp_path}/datasets"
        self.imag_path = f"{self.exp_path}/imags"
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.state_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.imag_path, exist_ok=True)

    def dump_descript(self, meta, params, status):
        experiment = {
            'meta': meta,
            'atmosphere': params._asdict(),
            'integrator': status,
        }
        with open(f"{self.exp_path}/descript.toml", "wb") as f:
            tomli_w.dump(experiment, f)

    def load_descript(self, exp_name="test"):
        exp_path = f"{self.path}/{exp_name}"
        with open(f"{exp_path}/descript.toml", "rb") as f:
            experiment = tomli.load(f)
        return experiment

    def dump_state(self, state, step=None):
        descript = self.load_descript(self.exp_name)
        base_path = f"{self.exp_path}/state.h5"
        with h5py.File(base_path, "w") as f:
            f.create_dataset("field", data=state.field)
            f.attrs["time"] = state.time
            f.attrs["descript"] = json.dumps(descript)

        if step is not None:
            step_path = f"{self.state_path}/state_{step}.h5"
            shutil.copy(base_path, step_path)

    def load_state(self, step=None, parse_descript=False):
        path = f"{self.exp_path}/state.h5"
        if step is not None:
            path = f"{self.state_path}/state_{step}.h5"

        with h5py.File(path, "r") as f:
            field = f["field"][:]
            time = f.attrs["time"]
            descript = json.loads(f.attrs["descript"])

        if parse_descript:
            return descript
        else:
            return time, field

    def dump_dataset(self, dataset, step=None):
        dataset = dataset.copy()
        descript = self.load_descript(self.exp_name)

        path = f"{self.exp_path}/climate.nc"
        if step is not None:
            path = f"{self.dataset_path}/atm_{step}.nc"

        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in dataset.data_vars
        }

        dataset.attrs['meta'] = json.dumps(dataset.attrs['meta'])
        dataset.attrs['atmosphere'] = json.dumps(dataset.attrs['atmosphere'])
        dataset.attrs['integrator'] = json.dumps(dataset.attrs['integrator'])

        dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)

    def load_dataset(self, step=None):
        path = f"{self.exp_path}/climate.nc"
        if step is not None:
            path = f"{self.dataset_path}/atm_{step}.nc"
        
        with xr.open_dataset(path) as ds:
            dataset = ds.load() 

            dataset.attrs['meta'] = json.loads(dataset.attrs['meta'])
            dataset.attrs['atmosphere'] = json.loads(dataset.attrs['atmosphere'])
            dataset.attrs['integrator'] = json.loads(dataset.attrs['integrator'])
            
            return dataset

    def state_dataset(self, state, datasets):
        descript = self.load_descript(self.exp_name)
        ds_state = datasets(state.time, state.field, descript)
        return ds_state

    def full_dataset(self, state_datasets):
        ds_full = xr.concat(state_datasets, dim="t")
        return ds_full