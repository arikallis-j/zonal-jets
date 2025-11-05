import jax.numpy as jnp
from jax import lax, jit, pmap, random
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import json, os

from .funcs import fft2, ifft2
from .consts import PI

key = random.key(42)

class Atmosphere(eqx.Module):
    dt: float = 0.05
    beta: float = 1.0
    mu: float = 0.018
    nu: float = 1e-14
    p: int = 1

    kf: float  = 32.0
    dk: float = 1.0
    f_amp: float = 1e-6
    tau: float = 1e-6

    def psi_hat(self, zeta_hat, k2):
        return -zeta_hat / k2

    def velocity_hat(self, psi_hat, iky, ikx):
        ux_hat = - iky * psi_hat
        uy_hat = + ikx * psi_hat
        return ux_hat, uy_hat

    def velocity(self, zeta_hat, k2, iky, ikx):
        psi_hat = self.psi_hat(zeta_hat, k2)
        ux_hat, uy_hat = self.velocity_hat(psi_hat, iky, ikx)
        ux = jnp.real(ifft2(ux_hat))
        uy = jnp.real(ifft2(uy_hat))
        return ux, uy
    
    def mask_forcing(self, k2):
        k = jnp.sqrt(k2)
        return jnp.where((k >= (self.kf-self.dk)) & (k <= (self.kf+self.dk)), 1.0, 0.0)

    def random_complex_gaussian(self, shape):
        key_r, key_i = random.split(key)
        real = random.normal(key_r, shape)
        imag = random.normal(key_i, shape)
        return real + 1j * imag    

    def make_white_forcing(self, k2, mask):
        band = self.mask_forcing(k2) * mask
        noise = self.random_complex_gaussian(k2.shape)
        noise = noise * band
        
        power = jnp.sum(jnp.abs(noise)**2)
        scale = jnp.where(power > 0, jnp.sqrt(self.f_amp / (power + 1e-16)), 0.0)
        return noise * scale
    

    @eqx.filter_jit
    def rhs(self, zeta_hat, k2, iky, ikx, mask):
        # деалясинг в спектре, чтобы исключить алиасинг при мультипликациях
        zeta_hat_dealiased = zeta_hat * mask

        # физ пространство
        # zeta = jnp.real(ifft2(zeta_hat_dealiased))
        psi_hat = self.psi_hat(zeta_hat_dealiased, k2)
        _, uy_hat = self.velocity_hat(psi_hat, iky, ikx)

        # скорость из вихря
        ux, uy = self.velocity(zeta_hat_dealiased, k2, iky, ikx)

        # производные вихря в физ. пространстве
        dzeta_dx = jnp.real(ifft2(ikx * zeta_hat_dealiased))
        dzeta_dy = jnp.real(ifft2(iky * zeta_hat_dealiased))

        # нелинейное слагаемое u·grad ω (в физическом пространстве)
        adv = ux * dzeta_dx + uy * dzeta_dy

        # перевести адвекцию в спектр
        adv_hat = fft2(adv)

        # кориолисово слагаемое
        coriolis_hat = self.beta * uy_hat

        # forcing
        _f = ifft2(self.make_white_forcing(k2,mask))
        f_hat = fft2(_f - _f.mean())

        # диффузионный член ν ∇^2 ω в спектре: -ν k^2 ω_hat
        diff_hat = - self.mu * zeta_hat_dealiased - self.nu * k2**self.p * zeta_hat_dealiased

        # dω_hat/dt = - fft(u·∇ω) - coriolis_hat + diff_hat + f_hat
        return -adv_hat - coriolis_hat + diff_hat + f_hat


class Grid:
    def __init__(self, N=10, L=2*PI, dealias=True):
        self.N = N
        self.L = L
        self.dealias = dealias

        self._create_coordinates()
        self._create_frequencies()
        self._get_dealiasing()

    def calc(self, atm, steps):
        atm.rhs(fft2(self.zeta), self.k2, self.iky, self.ikx, self.mask)
        @jit
        def rk4_step(zeta_hat):
            r1 = atm.rhs(zeta_hat, self.k2, self.iky, self.ikx, self.mask)
            r2 = atm.rhs(zeta_hat + 0.5 * atm.dt * r1, self.k2, self.iky, self.ikx, self.mask)
            r3 = atm.rhs(zeta_hat + 0.5 * atm.dt * r2, self.k2, self.iky, self.ikx, self.mask)
            r4 = atm.rhs(zeta_hat + atm.dt * r3, self.k2, self.iky, self.ikx, self.mask)
            return zeta_hat + atm.dt * (r1 + 2*r2 + 2*r3 + r4) / 6.0
            
        @jit
        def time_step(n, zeta_hat):
            return rk4_step(zeta_hat)

        @jit
        def integrate(zeta_hat_0, steps):
            return lax.fori_loop(0, steps, time_step, zeta_hat_0)
        
        file_path = "data/zeta.json"
        with open(file_path, 'r') as f:
            self.zeta = jnp.array(json.load(f))
            
        file_path = "data/T.json"
        with open(file_path, 'r') as f:
            self.T = json.load(f)

        zeta_hat = fft2(self.zeta)
        zeta_hat = integrate(zeta_hat, steps)

        self.zeta =  jnp.real(ifft2(zeta_hat))
        self.u, self.v = atm.velocity(zeta_hat, self.k2, self.iky, self.ikx)
        self.U = jnp.sqrt(self.u**2 + self.v**2)
        zeta_hat.block_until_ready()
        
        self.U_mean = jnp.mean(self.U)
        self.kR = (atm.beta/(2*self.U_mean))**(1/2)
        self.LR = 1 / self.kR 
        self.E_tot = self.U_mean**2/2
        self.T += atm.dt * steps 
        self.epsilon = self.E_tot/self.T

        self.n_R = self.kR
        self.n_beta = 0.5 * (atm.beta**3/self.epsilon)**(1/5)
        self.R_beta = self.n_beta/self.n_R
        
        file_path = "data/zeta.json"
        with open(file_path, 'w') as f:
            json.dump(self.zeta.tolist(), f, indent=4)

        file_path = f"data/T.json"
        with open(file_path, 'w') as f:
            json.dump(self.T, f, indent=4)

    
    
    def plot_zeta(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.zeta), levels=60)
        plt.colorbar()
        plt.title("Vorticity")
        if save:
            plt.savefig('zeta.png')
        if show:
            plt.show()

    def plot_U(self, show=True, save=False):
        plt.figure(figsize=(6,5))
        plt.contourf(np.array(self.X), np.array(self.Y), np.array(self.U), levels=60)
        plt.colorbar()
        plt.title("U(x,y)")
        if save:
            plt.savefig('U.png')
        if show:
            plt.show()

    def _create_coordinates(self):
        x = jnp.linspace(0, self.L, self.N, endpoint=False)
        y = jnp.linspace(0, self.L, self.N, endpoint=False)
        self.dy = self.L / self.N
        self.X, self.Y = jnp.meshgrid(x, y, indexing='ij')
        self.zeta = jnp.zeros((self.N, self.N))
        self.T = 0

        os.makedirs(f"data", exist_ok=True)
        file_path = f"data/zeta.json"
        with open(file_path, 'w') as f:
            json.dump(self.zeta.tolist(), f, indent=4)

        file_path = f"data/T.json"
        with open(file_path, 'w') as f:
            json.dump(self.T, f, indent=4)

    def _create_frequencies(self):
        self.k =  2 * PI * jnp.fft.fftfreq(self.N, d=(self.L / self.N))
        self.kx = self.k[:, None]
        self.ky = self.k[None, :]
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        
        k2 = self.kx**2 + self.ky**2
        self.k2 = jnp.where(k2 == 0.0, 1.0, k2) 
        

    def _get_dealiasing(self):
        if self.dealias:
            kxmax = jnp.max(jnp.abs(self.k))
            cutoff = 2.0/3.0 * kxmax
            Kx, Ky = jnp.meshgrid(jnp.abs(self.k), jnp.abs(self.k), indexing='ij')
            mask = jnp.where((Kx < cutoff) & (Ky < cutoff), 1.0, 0.0)
        else:
            mask = jnp.ones((self.N, self.N))
        
        self.mask = mask
        
