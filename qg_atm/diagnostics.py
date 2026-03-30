import xarray as xr
import numpy as np

def fft_phys(x, N):
    return 1/N**2 * np.fft.fft2(x)

def ifft_phys(x_hat, N):
    return N**2 * np.real(np.fft.ifft2(x_hat))

def pad_spectrum(a, N, N_pad, T):
    n_p = + (N+1)//2
    n_m = - (N)//2 
    a_pad = np.zeros((T, N_pad, N_pad), dtype=np.complex128)

    a_pad[:, :n_p, :n_p] = a[:, :n_p, :n_p]
    a_pad[:, n_m:, n_m:] = a[:, n_m:, n_m:]
    a_pad[:, :n_p, n_m:] = a[:, :n_p, n_m:]
    a_pad[:, n_m:, :n_p] = a[:, n_m:, :n_p] 

    return a_pad

def crop_spectrum(a_pad, N, T):
    n_p = + (N+1)//2
    n_m = - (N)//2 
    a = np.zeros((T, N, N), dtype=np.complex128)

    a[:, :n_p, :n_p] = a_pad[:, :n_p, :n_p] 
    a[:, n_m:, n_m:] = a_pad[:, n_m:, n_m:]
    a[:, :n_p, n_m:] = a_pad[:, :n_p, n_m:]
    a[:, n_m:, :n_p] = a_pad[:, n_m:, :n_p]

    return a

def J_adv(q_hat, psi_hat, ikx, iky, N, N_pad, T):
    ux_hat, uy_hat = - iky * psi_hat, + ikx * psi_hat
    dx_q_hat, dy_q_hat = ikx * q_hat, iky * q_hat

    ux_hat_pad, uy_hat_pad = pad_spectrum(ux_hat, N, N_pad, T), pad_spectrum(uy_hat, N, N_pad, T)
    dx_q_hat_pad, dy_q_hat_pad = pad_spectrum(dx_q_hat, N, N_pad, T), pad_spectrum(dy_q_hat, N, N_pad, T)
    ux_pad, uy_pad = ifft_phys(ux_hat_pad, N_pad), ifft_phys(uy_hat_pad, N_pad)
    dx_q_pad, dy_q_pad = ifft_phys(dx_q_hat_pad, N_pad), ifft_phys(dy_q_hat_pad, N_pad)
    j_adv_pad = ux_pad * dx_q_pad + uy_pad * dy_q_pad
    j_adv_hat_pad = fft_phys(j_adv_pad, N_pad)
    j_adv_hat = crop_spectrum(j_adv_hat_pad, N, T)

    return j_adv_hat

def spectrum(x_hat, k1, k_abs, T):
    x_spec = np.zeros((T, len(k_abs)))
    k_bin = np.floor(k1).astype(int)
    k_flat = k_bin.ravel()
    for t in range(T):
        x_spec[t] = np.bincount(k_flat, weights=x_hat[t].ravel(), minlength=len(k_abs))
    
    return x_spec


class Diagnostics:
    def __init__(self):
        pass

    def velocity(self, ds):
        ux, uy = ds['ux'], ds['ux']
        return np.sqrt(ux**2 + uy**2)
    
    def vorticity(self, ds):
        psi, N = ds['psi'], ds.attrs['scale.N']
        kx, ky = np.meshgrid(ds['k'], ds['k'], indexing='ij')
        k2 = kx**2 + ky**2
        psi_hat = fft_phys(psi, N)
        omega = ifft_phys(- k2 * psi_hat, N)
        ds_omega = (ds['psi'] * 0 + omega)
        return ds_omega

    def energy(self, ds):
        u = self.velocity(ds)
        return 0.5 * np.square(u)

    def enstrophy(self, ds):
        omega = self.vorticity(ds)
        return 0.5 * np.square(omega)

    def mean_velocity(self, ds):
        u = self.velocity(ds)
        return u.mean(axis=(1,2))

    def mean_vorticity(self, ds):
        omega = self.vorticity(ds)
        return omega.mean(axis=(1,2))

    def mean_energy(self, ds):
        e = self.energy(ds)
        return e.mean(axis=(1,2))

    def mean_enstrophy(self, ds):
        z = self.enstrophy(ds)
        return z.mean(axis=(1,2))

    def rms_velocity(self, ds):
        u = self.velocity(ds)
        return np.sqrt(np.square(u).mean())

    def rms_vorticity(self, ds):
        omega = self.vorticity(ds)
        return np.sqrt(np.square(omega).mean())

    def energy_spectrum(self, ds):
        psi, N, T = ds['psi'], ds.attrs['scale.N'], len(ds['t'])
        kx, ky = np.meshgrid(ds['k'], ds['k'], indexing='ij')
        k1, k_abs = np.sqrt(kx**2 + ky**2), ds['k_abs']
        
        psi_hat = fft_phys(psi, N)
        E_2d = 0.5 * k1**2 * (np.abs(psi_hat))**2

        return ("t", "k_abs"), spectrum(E_2d, k1, k_abs, T)

    def enstrophy_spectrum(self, ds):
        psi, N, T = ds['psi'], ds.attrs['scale.N'], len(ds['t'])
        kx, ky = np.meshgrid(ds['k'], ds['k'], indexing='ij')
        k1, k_abs = np.sqrt(kx**2 + ky**2), ds['k_abs']
        
        psi_hat = fft_phys(psi, N)
        Z_2d = 0.5 * (k1)**4 * (np.abs(psi_hat))**2

        return ("t", "k_abs"), spectrum(Z_2d, k1, k_abs, T)

    def energy_flow(self, ds):
        N, T = ds.attrs['scale.N'], len(ds['t'])
        q, psi = ds['q'], ds['psi']
        kx, ky = np.meshgrid(ds['k'], ds['k'], indexing='ij')
        k1, k_abs = np.sqrt(kx**2 + ky**2), ds['k_abs']
        N_pad = 3 * int(np.max(np.abs(ds['k'])))
        ikx, iky = 1j * kx,  1j * ky
        
        q_hat, psi_hat = fft_phys(q, N), fft_phys(psi, N)
        J_hat = J_adv(q_hat, psi_hat, ikx, iky, N, N_pad, T)
        T_e_2d = - np.real(np.conj(psi_hat) * J_hat)
        T_e_k = spectrum(T_e_2d, k1, k_abs, T)
        Pi_E = - np.cumsum(T_e_k, axis=1)
        
        return ("t", "k_abs"), Pi_E

    def enstrophy_flow(self, ds):
        N, T = ds.attrs['scale.N'], len(ds['t'])
        q, psi = ds['q'], ds['psi']
        kx, ky = np.meshgrid(ds['k'], ds['k'], indexing='ij')
        k1, k_abs = np.sqrt(kx**2 + ky**2), ds['k_abs']
        N_pad = 3 * int(np.max(np.abs(ds['k'])))
        ikx, iky = 1j * kx,  1j * ky
        
        q_hat, psi_hat = fft_phys(q, N), fft_phys(psi, N)
        J_hat = J_adv(q_hat, psi_hat, ikx, iky, N, N_pad, T)
        T_z_2d = - np.real(np.conj(q_hat) * J_hat)
        T_z_k = spectrum(T_z_2d, k1, k_abs, T)
        Pi_Z = - np.cumsum(T_z_k, axis=1)

        return ("t", "k_abs"), Pi_Z

 