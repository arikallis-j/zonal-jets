import numpy as np

def make_field(ds, key):
    x, y, field = ds['x'], ds['y'], ds[key].T
    return x, y, field

def make_spectrum(ds, key):
    k1, spec = ds['k1'], ds[key].T
    return k1, spec

def make_spectrum_field(ds, key):
    kx = np.fft.fftshift(ds['kx'])
    ky = np.fft.fftshift(ds['ky'])
    s_field = np.fft.fftshift(ds[key])
    return kx, ky, s_field

def make_vector_field(ds, key_x, key_y):
    x, y, field_x, field_y = ds['x'], ds['y'], ds[key_x].T, ds[key_y].T
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y, field_x, field_y

def make_mean_field(ds, key):
    t, mean_field = ds['t'], ds[key].T
    return t, mean_field