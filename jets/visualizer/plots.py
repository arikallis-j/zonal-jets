from matplotlib.colors import LogNorm

from .const import *
from .data import *

def plot_potential_vorticity(ds, axis, cmap='cmo.curl', levels=50, title='Potential Vorticity $q$'):
    x, y, q = make_field(ds, 'q')
    axis.contourf(x, y, q, cmap=cmap, levels=levels)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title(title)

def plot_absolute_velocity(ds, axis, cmap='cmo.speed', levels=50, title='Absolute Velocity $|u|$', stream=False):
    x, y, v = make_field(ds, 'v')
    axis.contourf(x, y, v, cmap=cmap, levels=levels)
    if stream:
        X, Y, ux, uy = make_vector_field(ds, 'ux', 'uy')
        axis.streamplot(X, Y, ux, uy, color=YELLOW, linewidth=2.0, density=0.4)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title(title)

def plot_x_velocity(ds, axis, cmap='cmo.balance', levels=50, title='X-axis Velocity $u_x$'):
    x, y, ux = make_field(ds, 'ux')
    axis.contourf(x, y, ux, cmap=cmap, levels=levels)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title(title)

def plot_y_velocity(ds, axis, cmap='cmo.balance', levels=50, title='Y-axis Velocity $u_y$'):
    x, y, uy = make_field(ds, 'uy')
    axis.contourf(x, y, uy, cmap=cmap, levels=levels)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title(title)

def plot_stream_function(ds, axis, cmap='cmo.deep', levels=50, title='Stream Function $\\psi$'):
    x, y, psi = make_field(ds, 'psi')
    axis.contourf(x, y, psi, cmap=cmap, levels=levels)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_title(title)

def plot_energy_spectrum(ds, axis, color=BLUE, title='Energy Spectrum $E(k)$', norm=False):
    k1, e = make_spectrum(ds, 'e')
    kf = ds.attrs['atmosphere']['kf']
    if norm:
        e = e/np.sum(e)
    axis.loglog(k1, e, color=color)
    axis.axvline(kf, color='r', linestyle='--')
    axis.set_xlabel('k')
    # axis.set_ylabel('E(k)')
    axis.set_title(title)
    axis.grid(True)

def plot_enstrophy_spectrum(ds, axis, color=GREEN, title='Enstrophy Spectrum $Z(k)$', norm=False):
    k1, z = make_spectrum(ds, 'z')
    kf = ds.attrs['atmosphere']['kf']
    if norm:
        z = z/np.sum(z)
    axis.loglog(k1, z, color=color)
    axis.axvline(kf, color='r', linestyle='--')
    axis.set_xlabel('k')
    # axis.set_ylabel('Z(k)')
    axis.set_title(title)
    axis.grid(True)

def plot_mean_energy(ds, axis, color=BLUE, title='Mean Energy $\\bar E(t)$'):
    t, e_mean = make_mean_field(ds, 'e_mean')
    axis.plot(t, e_mean, color=color, marker='o', markersize=2)
    axis.set_xlabel('t')
    # axis.set_ylabel('E(t)')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_title(title)
    axis.grid(True)

def plot_mean_enstrophy(ds, axis, color=GREEN, title='Mean Enstrophy $\\bar Z(t)$'):
    t, z_mean = make_mean_field(ds, 'z_mean')
    axis.plot(t, z_mean, color=color,  marker='o', markersize=2)
    axis.set_xlabel('t')
    # axis.set_ylabel('Z(t)')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_title(title)
    axis.grid(True)
    
def plot_energy_injection(ds, axis, color=BLUE, title='Energy Injection $\\dot E(t)$'):
    t, e_dot = make_mean_field(ds, 'e_dot')
    axis.plot(t, e_dot, color=color, marker='o', markersize=2)
    axis.axhline(np.mean(e_dot), color='r', linestyle='--')
    axis.set_xlabel('t')
    # axis.set_ylabel('E(t)')
    # axis.set_xscale('log')
    # axis.set_yscale('log')
    axis.set_title(title)
    axis.grid(True)

def plot_enstrophy_injection(ds, axis, color=GREEN, title='Enstrophy Injection $\\dot Z(t)$'):
    t, z_dot = make_mean_field(ds, 'z_dot')
    axis.plot(t, z_dot, color=color,  marker='o', markersize=2)
    axis.axhline(np.mean(z_dot), color='r', linestyle='--')
    axis.set_xlabel('t')
    # axis.set_ylabel('Z(t)')
    # axis.set_xscale('log')
    # axis.set_yscale('log')
    axis.set_title(title)
    axis.grid(True)

def plot_energy_field(ds, axis, cmap='cmo.thermal', title='Energy Spectrum $E(k_x, k_y)$'):
    kx, ky, E = make_spectrum_field(ds, 'E')
    axis.pcolormesh(kx, ky, E, norm=LogNorm(), cmap=cmap)
    axis.set_xlabel('Kx')
    axis.set_ylabel('Ky')
    axis.set_title(title)

def plot_energy_field(ds, axis, cmap='cmo.thermal', title='Enstrophy Spectrum $Z(k_x, k_y)$'):
    kx, ky, Z = make_spectrum_field(ds, 'Z')
    axis.pcolormesh(kx, ky, Z, norm=LogNorm(), cmap=cmap)
    axis.set_xlabel('Kx')
    axis.set_ylabel('Ky')
    axis.set_title(title)


plots = {
    'q': plot_potential_vorticity,
    'v': plot_absolute_velocity,
    'ux': plot_x_velocity,
    'uy': plot_y_velocity,
    'psi': plot_stream_function,
    'e': plot_energy_spectrum,
    'z': plot_enstrophy_spectrum,
    'e_mean': plot_mean_energy,
    'z_mean': plot_mean_enstrophy,
    'e_dot': plot_energy_injection,
    'z_dot': plot_enstrophy_injection,
    'E': plot_energy_field,
    'Z': plot_energy_field,
}