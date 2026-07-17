from jets import *

tau = 1

config = {
    'exp_name': 'test',
    'visual': {
        'fields': ['q', 'v'],
        'spectra': ['z', 'e'],
        'means' : ['z_mean', 'e_mean'],
    },
    'atmosphere': {
        'n_grid': 256, 
        'params': {
            'initial': 'zero', 
            'forcing': 'norm',
            'p': 2,
            'nu': 4e-8, 
            'r': 0.01,
            'epsilon': 1.0,
            'kf': 16,
            'dkf': 2,
        }
    },
    'integrator': {
        'method': 'rk4',
        'interval': 1 * tau,
        'n_points': 100 * tau,
    },
}

model = Driver()
model.setup(**config)
print(model.descript())

model.start()
model.run(10)
