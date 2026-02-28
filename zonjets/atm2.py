import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import xarray as xr
import os
from jax import lax, jit, random, devices
from jax.numpy import pi as PI

def fft2(x):
    return jnp.fft.fft2(x)

def ifft2(xhat):
    return jnp.fft.ifft2(xhat)

class Atmosphere:
    def __init__(self, name=None, N=256, h = 0.005, seed = 42, U = 10, L = 1e6, mu = 1e-8, nu = 1e-5, p = 1, beta = 1e-11, epsilon = 1e-10, kf=32, dk=1):
        self.name = name
        if not os.path.exists(f"data/{name}.nc"):
            self.U = U
            self.L = L
            
            self.mu = mu * (L/U)
            self.nu = nu * 1/(L**(2*p-1)*U)
            self.p = p
            self.beta = beta * (L**2/U)
            self.epsilon = epsilon * (L/U)**2
            self.kf = kf
            self.dk = dk
            
            self.l = 2 * PI
            self.N = N
            self.h = h
            self.seed = seed

            self._init_param()
            self._init_grid()

        else:
            self.ds = xr.open_dataset(f"data/{self.name}.nc")
            self._load_param()
            self._load_grid()
        
        
        self._init_terms()
        self._init_calc()

    def _load_param(self):
        self.U = self.ds['U'].values[0]
        self.L = self.ds['L'].values[0]

        self.p = self.ds['p'].values[0]
        self.mu = self.ds['mu'].values[0] 
        self.nu = self.ds['nu'].values[0]

        self.beta = self.ds['beta'].values[0]
        self.epsilon = self.ds['epsilon'].values[0]
        self.kf = self.ds['kf'].values[0]
        self.dk = self.ds['dk'].values[0]
        
        self.l = self.ds['l'].values[0]
        self.N = self.ds['N'].values[0]
        self.h = self.ds['h'].values[0]
        self.seed = self.ds['seed'].values[0]
        

        self.ds_param = xr.Dataset(
            data_vars={
                "U":       (("scale"),   [self.U],      {"units": "m/s",  "long_name": "Characteristic velocity"}),
                "L":       (("scale"),   [self.L],      {"units": "m",    "long_name": "Characteristic length"}),

                "mu":      (("physics"), [self.mu],     {"units": "1",    "long_name": "Dynamic viscosity"}),
                "nu":      (("physics"), [self.nu],     {"units": "1",    "long_name": "Kinematic viscosity"}),
                "p":      (("physics"), [self.p],     {"units": "1",    "long_name": "Kinematic viscosity power"}),
                "beta":    (("physics"), [self.beta],   {"units": "1",    "long_name": "Coriolis parameter"}),

                "epsilon": (("forcing"), [self.epsilon],{"units": "1",    "long_name": "Forcing amplitude"}),
                "kf":      (("forcing"), [self.kf],     {"units": "1",    "long_name": "Forcing wavenumber"}),
                "dk":      (("forcing"), [self.dk],     {"units": "1",    "long_name": "Forcing range"}),

                "N":       (("grid"),    [self.N],      {"units": "1",    "long_name": "Grid resolution"}),
                "l":       (("grid"),    [self.l],      {"units": "1",    "long_name": "Domain scale"}),
                "h":       (("grid"),    [self.h],      {"units": "1",    "long_name": "Time step"}),
                "seed":    (("grid"),    [self.seed],   {"units": "1",    "long_name": "Random seed"}),
            },
            attrs={"description": "Global atmospheric model parameters"}
        )
        self.key = random.key(self.seed)

    def _load_grid(self):
        self.x = jnp.linspace(0, self.l, self.N, endpoint=False)
        self.y = jnp.linspace(0, self.l, self.N, endpoint=False)

        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dxdy = self.dx * self.dy

        self.zeta = jnp.array(self.ds['zeta'].values[-1].T)
        self.t = self.ds['t'].values[-1]

        self.k =  2 * PI * jnp.fft.fftfreq(self.N, d=(self.l/self.N))
        self.dkdl =  self.N**2 / (2 * PI)**2
        self.kx = self.k[:, None]
        self.ky = self.k[None, :]
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        k2 = self.kx**2 + self.ky**2
        self.k2 = jnp.where(k2 == 0.0, 1.0, k2) 

        kxmax = jnp.max(jnp.abs(self.k))
        cutoff = 2.0/3.0 * kxmax
        self.Kx, self.Ky = jnp.meshgrid(jnp.abs(self.k), jnp.abs(self.k), indexing='ij')
        self.dealiasing = jnp.where((self.Kx < cutoff) & (self.Ky < cutoff), 1.0, 0.0)
        
        self.k1 = jnp.sqrt(k2)
        self.forcing = jnp.where((self.k1 >= (self.kf-self.dk)) & (self.k1  <= (self.kf+self.dk)), 1.0, 0.0)

        self.zeta_tilde = fft2(self.zeta) * self.dxdy
        self.ux_tilde = +self.iky/self.k2 * self.zeta_tilde
        self.uy_tilde = -self.ikx/self.k2 * self.zeta_tilde
        self.ux = jnp.real(ifft2(self.ux_tilde)) * self.dkdl
        self.uy = jnp.real(ifft2(self.uy_tilde)) * self.dkdl
        self.u = jnp.sqrt(self.ux**2 + self.uy**2)

        self.u_mean = jnp.mean(self.u)
        self.n_R = (self.beta/(2*self.u_mean))**(1/2)

        self.e_tot = self.u_mean**2/2
        self.Epsilon = self.e_tot/(self.t + 1e-15)
        self.n_beta = 0.5 * (self.beta**3/self.Epsilon)**(1/5)
        
        self.R_beta = self.n_beta/self.n_R

        self.ds_grid = xr.Dataset(
                data_vars={
                "X": (("x", "y"), np.array(self.X.T), {"units": "dimless", "long_name": "Length on X axis"}),
                "Y": (("x", "y"), np.array(self.Y.T), {"units": "dimless", "long_name": "Length on Y axis"}),
                "Kx": (("kx", "ky"), np.array(self.Kx.T), {"units": "dimless", "long_name": "Wavenumber on X axis"}),
                "Ky": (("kx", "ky"), np.array(self.Ky.T), {"units": "dimless", "long_name": "Wavenumber on Y axis"}),
                "k1": (("kx", "ky"), np.array(self.k1.T), {"units": "dimless", "long_name": "Absolute wavenumber"}),
                "k2": (("kx", "ky"), np.array(self.k2.T), {"units": "dimless", "long_name": "Square of wavenumber"}),
                "D": (("kx", "ky"), np.array(self.dealiasing.T), {"units": "dimless", "long_name": "Dealiasing condition"}),
                "F": (("kx", "ky"), np.array(self.forcing.T), {"units": "dimless", "long_name": "Forcing range"}),
            },
            coords={
                "x": self.x,
                "y": self.y,
                "kx": self.k,
                "ky": self.k,
            },
            attrs={"description": "Grid field"}
        )

        self.ds_field = xr.Dataset(
            data_vars={
                "zeta": (("t", "x", "y"), self.ds['zeta'].values, {"units": "dimless", "long_name": "Vorticity"}),
                "ux": (("t", "x", "y"), self.ds['ux'].values, {"units": "dimless", "long_name": "Velocity on X axis"}),
                "uy": (("t", "x", "y"), self.ds['uy'].values, {"units": "dimless", "long_name": "Velocity on Y axis"}),
                "u": (("t", "x", "y"), self.ds['u'].values, {"units": "dimless", "long_name": "Absolute velocity"}),
                "R_beta": (("t"), self.ds['R_beta'].values, {"units": "dimless", "long_name": "R beta"}),
            },
            coords={
                "x": self.x,
                "y": self.y,
                "t": self.ds['t'].values,
            },
            attrs={"description": "Physical field"}
        )

    def _init_param(self):
        self.ds_param = xr.Dataset(
            data_vars={
                "U":       (("scale"),   [self.U],      {"units": "m/s",  "long_name": "Characteristic velocity"}),
                "L":       (("scale"),   [self.L],      {"units": "m",    "long_name": "Characteristic length"}),

                "mu":      (("physics"), [self.mu],     {"units": "1",    "long_name": "Dynamic viscosity"}),
                "nu":      (("physics"), [self.nu],     {"units": "1",    "long_name": "Kinematic viscosity"}),
                "p":      (("physics"), [self.p],     {"units": "1",    "long_name": "Kinematic viscosity power"}),
                "beta":    (("physics"), [self.beta],   {"units": "1",    "long_name": "Coriolis parameter"}),

                "epsilon": (("forcing"), [self.epsilon],{"units": "1",    "long_name": "Forcing amplitude"}),
                "kf":      (("forcing"), [self.kf],     {"units": "1",    "long_name": "Forcing wavenumber"}),
                "dk":      (("forcing"), [self.dk],     {"units": "1",    "long_name": "Forcing range"}),

                "N":       (("grid"),    [self.N],      {"units": "1",    "long_name": "Grid resolution"}),
                "l":       (("grid"),    [self.l],      {"units": "1",    "long_name": "Domain scale"}),
                "h":       (("grid"),    [self.h],      {"units": "1",    "long_name": "Time step"}),
                "seed":    (("grid"),    [self.seed],   {"units": "1",    "long_name": "Random seed"}),
            },
            attrs={"description": "Global atmospheric model parameters"}
        )
        self.key = random.key(self.seed)

    def _init_grid(self):
        self.x = jnp.linspace(0, self.l, self.N, endpoint=False)
        self.y = jnp.linspace(0, self.l, self.N, endpoint=False)

        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dxdy = self.dx * self.dy

        self.zeta = 2 * random.uniform(self.key, (self.N, self.N)) - 1 
        #self.zeta = jnp.zeros((self.N, self.N))
        self.t = 0

        self.k =  2 * PI * jnp.fft.fftfreq(self.N, d=(self.l/self.N))
        self.dkdl =  self.N**2 / (2 * PI)**2
        self.kx = self.k[:, None]
        self.ky = self.k[None, :]
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        k2 = self.kx**2 + self.ky**2
        self.k2 = jnp.where(k2 == 0.0, 1.0, k2) 

        kxmax = jnp.max(jnp.abs(self.k))
        cutoff = 2.0/3.0 * kxmax
        self.Kx, self.Ky = jnp.meshgrid(jnp.abs(self.k), jnp.abs(self.k), indexing='ij')
        self.dealiasing = jnp.where((self.Kx < cutoff) & (self.Ky < cutoff), 1.0, 0.0)
        
        self.k1 = jnp.sqrt(k2)
        self.forcing = jnp.where((self.k1 >= (self.kf-self.dk)) & (self.k1  <= (self.kf+self.dk)), 1.0, 0.0)

        eps = 1e-36
        s = 4
        k_abs = jnp.abs(self.k1)
        kmax = k_abs.max()
        k_crit = 0
        alpha = -np.log(eps)
        self.sigma = np.where(k_abs - k_crit > 0, np.exp(- alpha * ((k_abs - k_crit)/kmax)**s), 1)

        self.zeta_tilde = fft2(self.zeta) * self.dxdy
        self.ux_tilde = +self.iky/self.k2 * self.zeta_tilde
        self.uy_tilde = -self.ikx/self.k2 * self.zeta_tilde
        self.ux = jnp.real(ifft2(self.ux_tilde)) * self.dkdl
        self.uy = jnp.real(ifft2(self.uy_tilde)) * self.dkdl
        self.u = jnp.sqrt(self.ux**2 + self.uy**2)

        self.u_mean = jnp.mean(self.u)
        self.n_R = (self.beta/(2*self.u_mean))**(1/2)

        self.e_tot = self.u_mean**2/2
        self.Epsilon = self.e_tot/(self.t + 1e-15)
        self.n_beta = 0.5 * (self.beta**3/self.Epsilon)**(1/5)
        
        self.R_beta = self.n_beta/self.n_R

        self.ds_grid = xr.Dataset(
                data_vars={
                "X": (("x", "y"), np.array(self.X.T), {"units": "dimless", "long_name": "Length on X axis"}),
                "Y": (("x", "y"), np.array(self.Y.T), {"units": "dimless", "long_name": "Length on Y axis"}),
                "Kx": (("kx", "ky"), np.array(self.Kx.T), {"units": "dimless", "long_name": "Wavenumber on X axis"}),
                "Ky": (("kx", "ky"), np.array(self.Ky.T), {"units": "dimless", "long_name": "Wavenumber on Y axis"}),
                "k1": (("kx", "ky"), np.array(self.k1.T), {"units": "dimless", "long_name": "Absolute wavenumber"}),
                "k2": (("kx", "ky"), np.array(self.k2.T), {"units": "dimless", "long_name": "Square of wavenumber"}),
                "D": (("kx", "ky"), np.array(self.dealiasing.T), {"units": "dimless", "long_name": "Dealiasing condition"}),
                "F": (("kx", "ky"), np.array(self.forcing.T), {"units": "dimless", "long_name": "Forcing range"}),
            },
            coords={
                "x": self.x,
                "y": self.y,
                "kx": self.k,
                "ky": self.k,
            },
            attrs={"description": "Grid field"}
        )

        self.ds_field = xr.Dataset(
            data_vars={
                "zeta": (("x", "y"), np.array(self.zeta.T), {"units": "dimless", "long_name": "Vorticity"}),
                "ux": (("x", "y"), np.array(self.ux.T), {"units": "dimless", "long_name": "Velocity on X axis"}),
                "uy": (("x", "y"), np.array(self.uy.T), {"units": "dimless", "long_name": "Velocity on Y axis"}),
                "u": (("x", "y"), np.array(self.u.T), {"units": "dimless", "long_name": "Absolute velocity"}),
                "R_beta": (("t"), np.array([self.R_beta]), {"units": "dimless", "long_name": "R beta"}),
            },
            coords={
                "x": self.x,
                "y": self.y,
                "t": [self.t],
            },
            attrs={"description": "Physical field"}
        )

        # объединяем параметры и первый шаг
        os.makedirs("data", exist_ok=True)
        self.ds = xr.merge([self.ds_param, self.ds_grid, self.ds_field])
        self.ds.to_netcdf(f"data/{self.name}.nc", mode="w")
    
    def _init_terms(self):
        self.a = - self.mu - (-1)**(self.p + 1) * self.nu * self.k2**self.p + self.ikx/self.k2 * self.beta
         
        @jit
        def b(zeta):
            zeta_d = self.dealiasing * zeta
            u = jnp.real(ifft2(+ self.iky/self.k2 * zeta_d)) * self.dkdl
            v = jnp.real(ifft2(- self.ikx/self.k2 * zeta_d)) * self.dkdl
            dzeta_dx = jnp.real(ifft2(self.ikx * zeta_d)) * self.dkdl
            dzeta_dy = jnp.real(ifft2(self.iky * zeta_d)) * self.dkdl 
            adv = u * dzeta_dx + v * dzeta_dy
            A = fft2(adv) * self.dxdy

            key_r, key_i = random.split(self.key)
            real = random.normal(key_r, zeta_d.shape)
            imag = random.normal(key_i, zeta_d.shape)
            xi_tilde = self.dealiasing * self.forcing * (- self.k2 * (real + 1j * imag))
            xi = ifft2(xi_tilde)
            E = self.epsilon * fft2(xi - xi.mean())

            N = - A #+ E

            return N
        
        self.b = b

        self.phi = jnp.exp(self.a * self.h/2)
        self.phi2 = self.phi**2

    def _init_calc(self):
        @jit
        def rk4(z):
            d1 = self.b(z)
            d2 = self.b((z + self.h/2*d1) * self.phi) / self.phi
            d3 = self.b((z + self.h/2*d2) * self.phi) / self.phi
            d4 = self.b((z + self.h*d3) * self.phi2) / self.phi2
            return self.sigma * self.phi2 * (z + self.h/6 * (d1 + 2*d2 + 2*d3 + d4))
        
        @jit
        def time_step(n, zeta_hat):
            return rk4(zeta_hat)

        @jit
        def integrate(zeta_hat_0, steps):
            return lax.fori_loop(0, steps, time_step, zeta_hat_0)

        self.integrate = integrate

    def calc(self, steps=10):
        self.zeta_tilde = fft2(self.zeta) * self.dxdy

        self.zeta_tilde = self.integrate(self.zeta_tilde, steps)
        self.zeta = jnp.real(ifft2(self.zeta_tilde)) * self.dkdl
        self.ux = jnp.real(ifft2(+self.iky/self.k2 * self.zeta_tilde * self.dealiasing)) * self.dkdl
        self.uy = jnp.real(ifft2(-self.ikx/self.k2 * self.zeta_tilde * self.dealiasing)) * self.dkdl
        self.u = jnp.sqrt(self.ux**2 + self.uy**2)

        self.u_mean = jnp.mean(self.u)
        self.n_R = (self.beta/(2*self.u_mean))**(1/2)

        self.e_tot = self.u_mean**2/2
        self.t += self.h * steps
        self.Epsilon = self.e_tot/self.t
        self.n_beta = 0.5 * (self.beta**3/self.Epsilon)**(1/5)
        
        self.R_beta = self.n_beta/self.n_R

        self.ux_tilde = jnp.fft.fft(self.ux, axis=0) * self.dx
        self.uy_tilde = jnp.fft.fft(self.uy, axis=0) * self.dx
        self.zet_tilde = jnp.fft.fft(self.zeta, axis=0) * self.dx
        self.e_tilde = (self.ux_tilde**2 + self.uy_tilde**2) * 1/2

        self.z_tilde = (self.zet_tilde**2) * 1/2
        self.rho_e = 1/(2*PI)**2 * jnp.abs(self.e_tilde)**2
        self.rho_z =  1/(2*PI)**2 * jnp.abs(self.z_tilde)**2
        
        nbins=50
        bins = np.linspace(0, self.kx.max(), nbins+1)
        k_center = 0.5*(bins[:-1] + bins[1:])
        E_k = np.zeros(nbins)
        Z_k = np.zeros(nbins)
        for i in range(nbins):
            mask = (self.k1 >= bins[i]) & (self.k1 < bins[i+1])
            E_k[i] = np.array(self.rho_e)[mask].sum()
            Z_k[i] = np.array(self.rho_z)[mask].sum()

        E_k = self.rho_e.mean(axis=0)
        Z_k = self.rho_z.mean(axis=0)

        kk = self.kx.squeeze(axis=1)

        mask = kk >= 0
        E_k = E_k[mask]
        Z_k = Z_k[mask]
        kk = kk[mask]
    
        self.e_k = np.array(E_k)
        self.z_k = np.array(Z_k)
        self.kk = kk


        ds_step = xr.Dataset(
            data_vars={
                "zeta": (("x", "y"), np.array(self.zeta.T), {"units": "dimless", "long_name": "Vorticity"}),
                "ux": (("x", "y"), np.array(self.ux.T), {"units": "dimless", "long_name": "Velocity on X axis"}),
                "uy": (("x", "y"), np.array(self.uy.T), {"units": "dimless", "long_name": "Velocity on Y axis"}),
                "u": (("x", "y"), np.array(self.u.T), {"units": "dimless", "long_name": "Absolute velocity"}),
                "R_beta": (("t"), np.array([self.R_beta]), {"units": "dimless", "long_name": "R beta"}),
            },
            coords={
                "x": self.x,
                "y": self.y,
                "t": [self.t],
            },
        )

        # самое важное: правильный concat
        self.ds_field = xr.concat([self.ds_field, ds_step], dim="t")
        self.ds = xr.merge([self.ds_param, self.ds_grid, self.ds_field])
        self.ds.to_netcdf(f"data/{self.name}.nc", mode="w")

    def animate(self, field_name, save=False, fps=30):
        # достаём данные из xarray
        field = self.ds[field_name].values  # форма: (Nt, Nx, Ny)
        Nt = field.shape[0]

        # Рисуем фигуру
        fig, ax = plt.subplots(figsize=(6,5))

        # фиксируем цветовую шкалу — иначе картинка будет «прыгать»
        vmax = np.abs(field).max()
        vmin = np.abs(field).min()
        if field_name != 'u':
            vmin = -vmax

        im = ax.imshow(field[0], origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax)

        ax.set_title("t = 0 | R_beta = 0")
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_data(field[frame])
            ax.set_title(f"t = {float(self.ds['t'][frame]):.2f} | R_beta = {float(self.ds['R_beta'][frame]):.2f}")
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=Nt, interval=100, blit=True
        )

        if save:
            ani.save(f"{field_name}_animation.mp4", writer="ffmpeg", fps=fps)
        
        ani_html = HTML(ani.to_html5_video())
        plt.close(fig)

        return ani_html
    
    def plot(self, field_name, show=True, save=False):
        # достаём данные из xarray
        field = self.ds[field_name].values  # форма: (Nt, Nx, Ny)
        Nt = field.shape[0]

        # Рисуем фигуру
        fig, ax = plt.subplots(figsize=(6,5))

        # фиксируем цветовую шкалу — иначе картинка будет «прыгать»
        vmax = np.abs(field).max()
        vmin = np.abs(field).min()
        if field_name != 'u':
            vmin = -vmax

        im = ax.imshow(field[0], origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax)

        ax.set_title(f'{field_name}')
        plt.colorbar(im, ax=ax)

        im.set_data(field[-1])
        ax.set_title(f"t = {float(self.ds['t'][-1]):.2f} | R_beta = {float(self.ds['R_beta'][-1]):.2f}")
        
        if save:
            plt.savefig(f'{field_name}.png')
        if show:
            plt.show()
    
    
    def plot_zeta(self, show=True, save=False, levels=60):
        
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.zeta), levels=levels, cmap='RdBu_r')
        plt.colorbar()
        plt.title("Vorticity")
        if save:
            plt.savefig('zeta.png')
        if show:
            plt.show()

    def plot_U(self, show=True, save=False, levels=60):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.u), levels=levels, cmap='RdBu_r')
        plt.colorbar()
        plt.title("U(x,y)")
        if save:
            plt.savefig('U.png')
        if show:
            plt.show()
    
    def plot_Ux(self, show=True, save=False, levels=60):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.ux), levels=levels, cmap='RdBu_r')
        plt.colorbar()
        plt.title("u(x,y)")
        if save:
            plt.savefig('Ux.png')
        if show:
            plt.show()
    
    def plot_Uy(self, show=True, save=False, levels=60):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.uy), levels=levels, cmap='seismic')
        plt.colorbar()
        plt.title("v(x,y)")
        if save:
            plt.savefig('Uy.png')
        if show:
            plt.show()

    def plot_Ek(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.loglog(self.kk, self.e_k)
        plt.title("E(k)")
        if save:
            plt.savefig('Ek.png')
        if show:
            plt.show()

    def plot_Zk(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.loglog(self.kk, self.z_k)
        plt.title("Z(k)")
        if save:
            plt.savefig('Zk.png')
        if show:
            plt.show()