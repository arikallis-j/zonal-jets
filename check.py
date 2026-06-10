"""
conda install -c conda-forge numpy matplotlib h5py xarray cmocean tqdm netCDF4 ffmpeg
pip install -U tomli_w "jax[cuda12]"
"""
import qg_atm as qg
import json


config = {
    "experiment": "stational-turbulence",
    "descript": {
        "N": 256,
        "M": 20,
        "initial": "random"
    },
    "print_state": "std",
    "first_state": "last",
    "pds_args": {
        "q_range": "end", 
        "u_range": "end", 
        "q_cmap":"balance"
    }
}

driver = qg.Driver(qg.Config(**config))
state = driver.dataload()
driver.print_state(state)