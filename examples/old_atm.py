import numpy as np
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

PI = np.pi
rng = np.random.default_rng(0)

@dataclass
class Config:
    name: str = "test"
    Nx: int = 256
    Ny: int = 256

    beta: float = 1.0
    mu: float = 0.018
    nu: float = 1e-14
    p: int = 4

    kf: float  = 32.0
    dkf: float = 1.0
    f_amp: float = 1e-6
    dt: float = 0.05

class Atmosphere:
    def __init__(self, cfg = None):
        if cfg is None:
            return None

        # configs
        self.name = cfg.name
        
        self.Nx, self.Ny = cfg.Nx, cfg.Ny
        self.size = (self.Nx, self.Ny)

        self.beta, self.mu = cfg.beta, cfg.mu
        self.nu, self.p = cfg.nu, cfg.p

        self.kf, self.dkf = cfg.kf, cfg.dkf, 
        self.f_amp, self.dt = cfg.f_amp, cfg.dt

        # physical grid
        self.x = np.linspace(0, 2*PI, self.Nx, endpoint=False)
        self.y = np.linspace(0, 2*PI, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # spectral grid
        self.kx = 2*PI*np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = 2*PI*np.fft.fftfreq(self.Ny, d=self.dy)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing="xy")
        self.K2 = self.Kx**2 + self.Ky**2
        self.K = np.sqrt(self.K2)

        # Деалайзинг по правилу 2/3
        kx_cut = 2/3 * np.max(np.abs(self.kx))
        ky_cut = 2/3 * np.max(np.abs(self.ky))
        self.dealias_mask = (np.abs(self.Kx) <= kx_cut) & (np.abs(self.Ky) <= ky_cut)

        # Форсирование в спектре
        self.force_mask = ((self.K >= self.kf - self.dkf) & (self.K <= self.kf + self.dkf)).astype(float)

        self.nonzero = self.K2 != 0

        # phisical fields
        self.zeta = np.zeros(self.size)
        self.psi = np.zeros(self.size)
        self.u = np.zeros(self.size)
        self.v = np.zeros(self.size)
        self.U = np.zeros(self.size)

        # spectral fields
        self.zeta_hat = np.zeros(self.size)
        self.L_hat = np.zeros(self.size, dtype=complex)
        self.L_hat[self.nonzero] = - self.mu - self.beta * 1j * self.Kx[self.nonzero]  / self.K2[self.nonzero]  - self.nu * self.K2[self.nonzero]**(self.p)
        self.L_hat[0,0] = - self.mu
        
        self.T = 0

    def calc(self, zeta_hat):
        psi_hat = np.zeros(self.size, dtype=complex)
        psi_hat[self.nonzero] = -zeta_hat[self.nonzero] / self.K2[self.nonzero]
        psi_hat[0,0] = 0.0
        u_hat = - 1j * self.Ky * psi_hat
        v_hat = + 1j * self.Kx * psi_hat
        zeta_x_hat =  + 1j * self.Kx * zeta_hat
        zeta_y_hat = + 1j * self.Ky * zeta_hat

        # Вычисляем силы
        f_hat = self._forcing()
        
        # вычисляем адвекцию
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        zeta_x =  np.real(np.fft.ifft2(zeta_x_hat))
        zeta_y =  np.real(np.fft.ifft2(zeta_y_hat))
        adv_phis = zeta_x*u + zeta_y*v
        adv_hat = np.fft.fft2(adv_phis)
        adv_hat = self._dealias(adv_hat, self.dealias_mask)

        # Вычисляем нелинейный член
        N_hat = - adv_hat + f_hat

        return N_hat

    def update(self, zeta_hat):
        self.zeta_hat = zeta_hat
        self.zeta = np.real(np.fft.ifft2(zeta_hat))

        psi_hat = np.zeros(self.size, dtype=complex)
        psi_hat[self.nonzero] = - zeta_hat[self.nonzero] / self.K2[self.nonzero]
        psi_hat[0,0] = 0.0
        u_hat = - 1j * self.Ky * psi_hat
        v_hat = + 1j * self.Kx * psi_hat

        self.psi = np.real(np.fft.ifft2(psi_hat))
        self.u = np.real(np.fft.ifft2(u_hat))
        self.v = np.real(np.fft.ifft2(v_hat))
        self.U = np.sqrt(self.u**2 + self.v**2)
        self.U_mean = np.mean(self.U)
        self.kR = (self.beta/(2*self.U_mean))**(1/2)
        self.LR = 1 / self.kR 
        self.E_tot = self.U_mean**2/2
        self.T += self.dt
        self.epsilon = self.E_tot/self.T

        self.n_R = self.kR
        self.n_beta = 0.5 * (self.beta**3/self.epsilon)**(1/5)
        self.R_beta = self.n_beta/self.n_R
        self.R_beta = 0.7 * (self.beta * self.U_mean**5/self.epsilon**2)**(1/10)


    def step(self, mod='etdrk4'):
        # ETD1
        if mod=='etd1':
            N_hat = self.calc(self.zeta_hat)
            E = np.exp(self.L_hat*self.dt)

            zeta_hat_new = E * self.zeta_hat + (E - 1)/self.L_hat * N_hat
            self.update(zeta_hat_new)

        # ETDRK2
        elif mod=='etdrk2':
            E = np.exp(self.L_hat*self.dt)
            E2 =  np.exp(self.L_hat*self.dt/2)

            N_hat = self.calc(self.zeta_hat)
            zeta_hat_1 = E2 * self.zeta_hat + (E2 - 1)/self.L_hat * N_hat
            N_hat_1 = self.calc(zeta_hat_1)
            zeta_hat_new = E * self.zeta_hat + (E - 1)/self.L_hat * N_hat_1
            
            self.update(zeta_hat_new)

        # ETDRK3
        elif mod=='etdrk3':
            E = np.exp(self.L_hat*self.dt)
            E2 =  np.exp(self.L_hat*self.dt/2)

            N_hat = self.calc(self.zeta_hat)
            zeta_hat_1 = E2 * self.zeta_hat + (E2 - 1)/self.L_hat * N_hat
            N_hat_1 = self.calc(zeta_hat_1)
            zeta_hat_2 = E2 * self.zeta_hat + (E2 - 1)/self.L_hat * N_hat_1
            N_hat_2 = self.calc(zeta_hat_2)
            zeta_hat_new = E * self.zeta_hat + (E - 1)/self.L_hat * (1/3*N_hat + 2/3*N_hat_2)
            
            self.update(zeta_hat_new)

        # ETDRK4
        elif mod=='etdrk4':
            E = np.exp(self.L_hat*self.dt)
            E2 =  np.exp(self.L_hat*self.dt/2)

            N_hat = self.calc(self.zeta_hat)
            zeta_hat_a = E2 * self.zeta_hat + (E2 - 1)/self.L_hat * N_hat
            N_hat_a = self.calc(zeta_hat_a)
            zeta_hat_b = E2 * self.zeta_hat + (E2 - 1)/self.L_hat * N_hat_a
            N_hat_b = self.calc(zeta_hat_b)
            zeta_hat_c = E * self.zeta_hat + (E - 1)/self.L_hat * (2*N_hat_b - N_hat)
            N_hat_c = self.calc(zeta_hat_c)
            zeta_hat_new = E * self.zeta_hat + (E - 1)/self.L_hat * (1/6*N_hat + 2/6*(N_hat_a + N_hat_b) + 1/6*N_hat_c)
            
            self.update(zeta_hat_new)
        else:
            print("Warning: error mode")
        
    def _forcing(self):
        re = rng.standard_normal((self.Ny, self.Nx))
        im = rng.standard_normal((self.Ny, self.Nx))
        fh = (re + 1j*im) * self.force_mask

        # Hermitian symmetry
        #fh = 0.5*(np.fft.fftshift(fh) + np.conj(np.flipud(np.fliplr(np.fft.fftshift(fh)))))
        #fh = np.fft.ifftshift(fh)
        fh = fh * self.f_amp * np.sqrt(self.dt)
        fh = self._dealias(fh, self.dealias_mask)
        return fh

    def _dealias(self, s_hat, mask):
        """Применяем деалайзинг по маске."""
        if mask is None:
            return s_hat
        s_hat_dealias = s_hat.copy()
        s_hat_dealias[~mask] = 0.0
        return s_hat_dealias
    
    def plot_x_grid(self):
        plt.figure()
        for xi in self.x:
            plt.plot([xi] * len(self.y), self.y, color="gray")  # вертикальные линии
        for yi in self.y:
            plt.plot(self.x, [yi] * len(self.x), color="gray")  # горизонтальные линии

        # plt.scatter(self.X, self.Y, color='grey')
        plt.title("XY-grid")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.show()

    def plot_k_grid(self):
        plt.figure()
        for xi in self.kx:
            plt.plot([xi] * len(self.ky), self.ky, color="gray")  # вертикальные линии
        for yi in self.ky:
            plt.plot(self.kx, [yi] * len(self.kx), color="gray")  # горизонтальные линии

        # plt.scatter(self.Kx, self.Ky, color='grey')
        plt.title("KxKy-grid")
        plt.xlabel("kx")
        plt.ylabel("ky")
        plt.axis("equal")
        plt.show()

    def plot_K(self):
        plt.figure()
        pcm = plt.pcolormesh(self.X, self.Y, self.K, shading="auto", cmap="viridis")
        plt.colorbar(pcm, label="|K|")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Module K")
        plt.axis("equal")
        plt.show()

    def plot_zeta(self):
        plt.figure()
        pcm = plt.pcolormesh(self.X, self.Y, self.zeta, shading="auto", cmap="viridis")
        plt.colorbar(pcm, label="zeta")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Zeta")
        plt.axis("equal")
        plt.show()

    def plot_U(self):
        plt.figure()
        pcm = plt.pcolormesh(self.X, self.Y, self.U, shading="auto", cmap="viridis")
        plt.colorbar(pcm, label="U")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("U")
        plt.axis("equal")
        plt.savefig('U.png')

    def plot_uv(self, step=2, arrow_frac=0.5):
        """
        Отображает поле скоростей:
        - фон: |U| (cmap='plasma')
        - стрелки одинаковой длины, помещаются в квадрат.
        step: прореживание стрелок по сетке
        arrow_frac: доля (0..1) между соседними узлами, которую займёт длина стрелки
        """

        plt.figure()
        pcm = plt.pcolormesh(self.X, self.Y, self.U, shading='auto', cmap='viridis')

        # прореженная сетка для стрелок
        Xs = self.X[::step, ::step]
        Ys = self.Y[::step, ::step]
        us = self.u[::step, ::step]
        vs = self.v[::step, ::step]

        mag = np.sqrt(us**2 + vs**2)
        mag[mag == 0] = 1.0
        usn = us / mag
        vsn = vs / mag
        dx_plot = (self.x[1] - self.x[0]) * step
        dy_plot = (self.y[1] - self.y[0]) * step
        arrow_length = min(dx_plot, dy_plot) * arrow_frac

        U_plot = usn * arrow_length
        V_plot = vsn * arrow_length

        # Рисуем стрелки в координатных единицах: scale_units='xy', scale=1
        plt.quiver(
            Xs, Ys, U_plot, V_plot,
            color='white',
            angles='xy', scale_units='xy', scale=1,
            pivot='mid', width=0.003, headwidth=3, headlength=4
        )

        plt.colorbar(pcm, label='|U|') 
        plt.title("Поле скоростей (стрелки одинаковой длины)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(self.y[0], self.y[-1])
        plt.tight_layout()
        plt.show()

    def animate_field(self, field='zeta', nsteps=100, filename="anim.mp4",
                    interval=50, fps=20, cmap="viridis", vmin=None, vmax=None,
                    step_show=1, use_pcolormesh=True):
        """
        Анимация для self.zeta или self.U с учётом нового интерфейса.
        """
        fig, ax = plt.subplots()
        data0 = getattr(self, field).copy()
        if vmin is None: vmin = np.nanmin(data0)
        if vmax is None: vmax = np.nanmax(data0)

        if use_pcolormesh:
            mesh = ax.pcolormesh(self.X, self.Y, data0, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            extent = [self.x[0], self.x[-1]+(self.x[1]-self.x[0]), self.y[0], self.y[-1]+(self.y[1]-self.y[0])]
            mesh = ax.imshow(data0, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{field} field")
        ax.set_aspect("equal")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        if field=='uv':
            sx = step_show
            Xs, Ys = self.X[::sx, ::sx], self.Y[::sx, ::sx]
            us, vs = self.u[::sx, ::sx], self.v[::sx, ::sx]
            mag = np.sqrt(us**2 + vs**2)
            mag[mag==0]=1
            usn, vsn = us/mag, vs/mag
            dx_plot, dy_plot = (self.x[1]-self.x[0])*sx, (self.y[1]-self.y[0])*sx
            arrow_length = min(dx_plot, dy_plot)*0.5
            quiv = ax.quiver(Xs, Ys, usn*arrow_length, vsn*arrow_length, color="white",
                            angles="xy", scale_units="xy", scale=1, pivot="mid", width=0.003)
        else:
            quiv = None

        def update(frame):
            nonlocal mesh, quiv
            self.step(mod='etd1')  # шаг интегрирования

            data = getattr(self, field)
            if use_pcolormesh:
                mesh.remove()
                mesh = ax.pcolormesh(self.X, self.Y, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                mesh.set_data(data)

            if field=='U' and quiv is not None:
                quiv.remove()
                us, vs = self.u[::sx, ::sx], self.v[::sx, ::sx]
                mag = np.sqrt(us**2 + vs**2)
                mag[mag==0]=1
                usn, vsn = us/mag, vs/mag
                quiv = ax.quiver(Xs, Ys, usn*arrow_length, vsn*arrow_length, color="white",
                                angles="xy", scale_units="xy", scale=1, pivot="mid", width=0.003)
            ax.set_title(f"{field}, step {frame+1}/{nsteps}")
            print(f"{field}, step {frame+1}/{nsteps}")
            return mesh, quiv if quiv else mesh

        cbar = fig.colorbar(mesh, ax=ax, label=field)

        anim = FuncAnimation(fig, update, frames=nsteps, interval=interval, blit=False)

        if filename.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=fps)
            anim.save(filename, writer=writer, dpi=150)
        elif filename.lower().endswith(".gif"):
            anim.save(filename, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Анимация сохранена в {filename}")


config = Config(Nx=256,Ny=256,
    beta = 1.0,
    mu = 1.4e-3,
    nu = 1e-9,
    p = 4,
    kf  = 32.0,
    dkf = 1.0,
    f_amp = 1e-3,
    dt = 0.01,
)
atm = Atmosphere(config)
N = 100
k = 0 
while True:
    atm.step()#mod='etdrk4'
    if atm.R_beta > 2:
        break
    print(f"step {k+1}, {atm.R_beta:.3f}")
    if k%100 == 0:
        atm.plot_U()
    k += 1

atm.plot_U()
atm.plot_uv()
# atm.animate_field(field='U')
