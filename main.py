from qg_atm import *
from test import test_all

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

descript = {
    'N': 256,
    'M': 20,
    'initial': 'random',
}

config = Config(
    experiment="stational-turbulence",
    descript=descript, 
    print_dataset="std", 
    pds_args={'q_range': "end", 'u_range': "end", 'q_cmap':"balance"},
)

driver = Driver(config)
result = driver.run(n_iter=100)
