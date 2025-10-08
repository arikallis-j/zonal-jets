import numpy as np

Nt, Nx, Ny = 10, 10, 10 
Lt, Lx, Ly = 1.0, 1.0, 1.0
SIZE = (Nx, Ny)

beta = 1.0
nu_hyper = 1e-5
mu = 0.1
hyper_order = 2


def apply_dealiasing_in_spectrum(s_hat, mask):
    """Обнуляет спектр вне маски (маска булева по kx,ky)."""
    if mask is None:
        return s_hat
    out = s_hat.copy()
    out[~mask] = 0.0
    return out

class Atmosphere:
    def __init__ (self, length = (Lx, Ly), size = SIZE, dealias=True):
        self.length = length
        self.size = size

        # сетка координат
        self.x = np.linspace(0, length[0], size[0], endpoint=False)
        self.y = np.linspace(0, length[1], size[1], endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
        # cетка обратных координат
        self.kx = 2*np.pi * np.fft.fftfreq(size[0], d=length[0]/size[0])
        self.ky = 2*np.pi * np.fft.fftfreq(size[1], d=length[1]/size[1])
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing='xy')
        self.K = (self.Kx**2 + self.Ky**2)**0.5

        # деалайзинг 2/3 rule: нули для |k| > 2/3*k_max
        self.dealias_mask = None
        if dealias:
            kx_max = np.max(np.abs(self.kx))
            ky_max = np.max(np.abs(self.ky))
            kx_cut = (2/3)*kx_max
            ky_cut = (2/3)*ky_max
            self.dealias_mask = (np.abs(self.Kx) <= kx_cut) & (np.abs(self.Ky) <= ky_cut)
            
        # основные переменные
        self.zeta = np.random.random(size)
        self.psi = np.zeros(size)
        self.u = np.zeros(size)
        self.v = np.zeros(size)
        self.U = np.zeros(size)

    def calc_psi(self, zeta):
        # расчет пуассона для функции тока
        mean_psi = 0.0
        zeta_hat = np.fft.fft2(zeta)
        psi_hat = np.zeros_like(zeta_hat, dtype=complex)
        nonzero = self.K != 0.0
        psi_hat[nonzero] = - zeta_hat[nonzero] / self.K[nonzero]**2
        psi_hat[0,0] = mean_psi * self.size[0] * self.size[1]
        self.psi = np.real(np.fft.ifft2(psi_hat))

        # расчет новых скоростей по функции тока
        psi_hat = np.fft.fft2(self.psi)
        dpsi_dx_hat = 1j * self.Kx * psi_hat
        dpsi_dy_hat = 1j * self.Ky * psi_hat
        dpsi_dx = np.real(np.fft.ifft2(dpsi_dx_hat))
        dpsi_dy = np.real(np.fft.ifft2(dpsi_dy_hat))
        self.u = - dpsi_dy
        self.v = + dpsi_dx

        # модуль скорости
        self.U = (self.u**2 + self.v**2)**0.5

    def calc_R(self, zeta):

        zeta_hat = np.fft.fft2(zeta)

        # спектральная производная
        dzeta_dx_hat = 1j * self.Kx * zeta_hat
        dzeta_dy_hat = 1j * self.Ky * zeta_hat
        dzeta_dx = np.real(np.fft.ifft2(dzeta_dx_hat))
        dzeta_dy = np.real(np.fft.ifft2(dzeta_dy_hat))

        # вычисление адвективного члена
        adv_term = self.u * dzeta_dx + self.v * dzeta_dy 
        adv_hat = np.fft.fft2(adv_term)
        adv_hat = apply_dealiasing_in_spectrum(adv_hat, self.dealias_mask)
        adv_phys = np.real(np.fft.ifft2(adv_hat))
        
        # источник beta*v
        source = beta * self.v

        # источник mu*zeta
        lin_diss = mu * zeta

        # диссипация в спектре
        diss_phys = 0.0
        if nu_hyper != 0.0:
            # в спектре: diss_hat = - nu * k^{2*hyper_order} * z_hat
            k_pow = self.K ** (2*hyper_order)
            diss_hat = - nu_hyper * k_pow * zeta_hat
            diss_hat = apply_dealiasing_in_spectrum(diss_hat, self.dealias_mask)  # по желанию
            diss_phys = np.real(np.fft.ifft2(diss_hat))
        R = - adv_phys - source - lin_diss + diss_phys


        return R

    def RK3(self, zeta, dt):
        """SSP Рунге-Кута 3 порядка одном шаг."""
        self.calc_psi(zeta)
        R1 = self.calc_R(zeta)
        zeta_1 = zeta + dt * R1

        R2 = self.calc_R(zeta_1)
        zeta_2 = 0.75* + 0.25*(zeta_1 + dt * R2)

        R3 = self.calc_R(zeta_2)
        z_next = (1.0/3.0)*zeta + (2.0/3.0)*(zeta_2 + dt * R3)
        return z_next

    def calc(self):
        for t in range(Nt):
            dt = Lt/Nt
            self.zeta = self.RK3(self.zeta, dt)


atm = Atmosphere()
atm.calc()
print(atm.U)