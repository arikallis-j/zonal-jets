"""
conda install -c conda-forge numpy matplotlib h5py xarray cmocean tqdm netCDF4 ffmpeg
pip install -U tomli_w "jax[cuda12]"
"""
import qg_atm as qg

config = {
    'experiment': "dissipation-free",
    'n_iter': 100,
    't_iter': 100,
    'descript': {
        'N': 256,
        'M': 10,
        'initial': 'random',
    },
    'first_state': "last",
    'print_dataset': "all",
    'print_state': "std",
    'pds_args': {
        'q_range': "end", 
        'u_range': "end", 
        'q_cmap':"balance"
    },
}

driver = qg.Driver(qg.Config(**config))
result = driver.run()
