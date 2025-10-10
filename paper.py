import numpy as np
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt

PI = np.pi
rng = np.random.default_rng(42)

@dataclass
class Config:
    name: str = "test"
    Ny: int = 256
    Nx: int = 256
    beta: float = 1.0
    nu: float = 1e-14
    mu: float = 0.01824
    hyper_order: int = 1
    kf: float  = 8.0
    dk: float = 1.0
    forc_amp: float = 1e-6
    dt: float = 0.005

def dealias(s_hat, mask):
    """Применяем деалайзинг по маске."""
    if mask is None:
        return s_hat
    s_hat_dealias = s_hat.copy()
    s_hat_dealias[~mask] = 0.0
    return s_hat_dealias

class Atmosphere:
    def __init__(self, cfg: Config):
        # Сетка
        self.Nx, self.Ny = cfg.Nx, cfg.Ny
        self.x = np.linspace(0, 2*PI, self.Nx, endpoint=False)
        self.y = np.linspace(0, 2*PI, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Параметры
        self.beta, self.nu, self.mu = cfg.beta, cfg.nu, cfg.mu
        self.hyper_order = cfg.hyper_order
        self.kf, self.dk = cfg.kf, cfg.dk
        self.forc_amp = cfg.forc_amp
        self.dt = cfg.dt

        # Преобразование Фурье
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
        self.force_mask = ((self.K >= self.kf - self.dk) & (self.K <= self.kf + self.dk)).astype(float)

        # Поля
        self.zeta = np.zeros((self.Ny, self.Nx))
        self.psi, self.u, self.v, self.U = self.calc_psi(self.zeta)
        self.R = self.calc_R(self.zeta, self.u, self.v)
        self.dt = cfg.dt

    def calc_psi(self, zeta):
        z_hat = np.fft.fft2(zeta)
        psi_hat = np.zeros_like(z_hat, dtype=complex)
        nonzero = self.K2 != 0
        psi_hat[nonzero] = - z_hat[nonzero] / self.K2[nonzero]
        psi = np.real(np.fft.ifft2(psi_hat))

        # скорости
        psi_hat = np.fft.fft2(psi)
        u = -np.real(np.fft.ifft2(1j*self.Ky*psi_hat))
        v =  np.real(np.fft.ifft2(1j*self.Kx*psi_hat))
        U = np.sqrt(u**2 + v**2)
        return psi, u, v, U

    def calc_f(self, dt, amp):
        """Случайное форсирование."""
        re = rng.standard_normal((self.Ny, self.Nx))
        im = rng.standard_normal((self.Ny, self.Nx))
        fh = (re + 1j*im) * self.force_mask
        # Hermitian symmetry
        fh = 0.5*(np.fft.fftshift(fh) + np.conj(np.flipud(np.fliplr(np.fft.fftshift(fh)))))
        fh = np.fft.ifftshift(fh) * amp * np.sqrt(dt)
        fh = dealias(fh, self.dealias_mask)
        return fh

    def calc_R(self, zeta, u, v):
        z_hat = np.fft.fft2(zeta)
        dzdx = np.real(np.fft.ifft2(1j*self.Kx*z_hat))
        dzdy = np.real(np.fft.ifft2(1j*self.Ky*z_hat))
        adv = u*dzdx + v*dzdy
        adv_hat = dealias(np.fft.fft2(adv), self.dealias_mask)
        adv_phys = np.real(np.fft.ifft2(adv_hat))

        diss_hat = - self.nu * self.K**(2*self.hyper_order) * z_hat
        diss_phys = np.real(np.fft.ifft2(dealias(diss_hat, self.dealias_mask)))

        force_phys = np.real(np.fft.ifft2(self.calc_f(self.dt, self.forc_amp)))

        R = -adv_phys - self.beta*v - self.mu*zeta + diss_phys + force_phys
        return R

    def calc_zeta(self, zeta, dt):
        """SSP-RK3 шаг."""
        psi, u, v, U = self.calc_psi(zeta)
        R1 = self.calc_R(zeta, u, v)
        z1 = zeta + dt*R1

        psi1, u1, v1, U1 = self.calc_psi(z1)
        R2 = self.calc_R(z1, u1, v1)
        z2 = 0.75*zeta + 0.25*(z1 + dt*R2)

        psi2, u2, v2, U2 = self.calc_psi(z2)
        R3 = self.calc_R(z2, u2, v2)
        z_next = (1/3)*zeta + (2/3)*(z2 + dt*R3)

        return psi2, u2, v2, U2, z_next

    def step(self):
        self.psi, self.u, self.v, self.U, self.zeta = self.calc_zeta(self.zeta, self.dt)
        return self.psi, self.u, self.v, self.U, self.zeta


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
        plt.show()

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

    def animate_zeta(self, dt, nsteps, filename="zeta_anim.mp4", interval=50, fps=20,
                  cmap="viridis", vmin=None, vmax=None, step_show=1, use_pcolormesh=True):
        """
        Сохранить анимацию поля |U| для nsteps шагов по dt.
        Параметры:
            dt        - временной шаг, который будет использовать calc_dt
            nsteps    - число кадров/шагов интегрирования
            filename  - куда сохранять (mp4 или gif)
            interval  - миллисекунды между кадрами при просмотре (не влияет на файл)
            fps       - fps для сохранения (для mp4)
            cmap      - цветовая карта (по умолчанию 'plasma')
            vmin/vmax - если задать, цветовая шкала фиксируется; иначе берётся по данным первого кадра
            step_show - показывать каждую step_show точку в pcolormesh/quiver прореживании
            use_pcolormesh - если True, используем pcolormesh; иначе imshow
        Примечание: для сохранения mp4 нужен ffmpeg установленный в системе.
        """

        # --- подготовка фигуры/оси ---
        fig, ax = plt.subplots()

        # вычислим начальное поле U (на текущем self.zeta)
        # убедимся, что self.U актуально (если объект только создан, оно уже есть)
        zeta0 = self.zeta.copy()

        # фиксируем цветовую шкалу, если не задана
        if vmin is None:
            vmin = float(np.nanmin(zeta0))
        if vmax is None:
            vmax = float(np.nanmax(zeta0))

        # отрисовка начального кадра
        if use_pcolormesh:
            mesh = ax.pcolormesh(self.X, self.Y, zeta0, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # imshow удобен для регулярных сеток, но тогда нужно правильно выставить extent
            extent = [self.x[0], self.x[-1] + (self.x[1]-self.x[0]), self.y[0], self.y[-1] + (self.y[1]-self.y[0])]
            mesh = ax.imshow(zeta0, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(mesh, ax=ax, label="zeta")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Модуль zeta")

        ax.set_aspect("equal")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        # --- функция обновления (каждый кадр делаем шаг интегрирования и обновляем данные) ---
        def update(frame):
            nonlocal mesh
            self.calc_dt(dt)

            if use_pcolormesh:
                mesh.remove()
                mesh = ax.pcolormesh(self.X, self.Y, self.zeta,
                                     shading="auto", cmap=cmap,
                                     vmin=vmin, vmax=vmax)
            else:
                mesh.set_data(self.zeta)

            ax.set_title(f"zeta, step {frame+1}/{nsteps}")
            print(f"step {frame+1}/{nsteps}")
            return mesh


        # --- создаём анимацию ---
        anim = FuncAnimation(fig, update, frames=nsteps, interval=interval, blit=False)

        # --- сохраняем ---
        # выбираем формат по расширению
        if filename.lower().endswith(".mp4"):
            # ffmpeg writer
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(filename, writer=writer, dpi=150)
            except Exception as e:
                plt.close(fig)
                raise RuntimeError("Не удалось сохранить mp4: проверьте, установлен ли ffmpeg. Ошибка: " + str(e))

        plt.close(fig)
        print(f"Анимация сохранена в {filename}")

    def animate_U(self, dt, nsteps, filename="U_anim.mp4", interval=50, fps=20,
                  cmap="viridis", vmin=None, vmax=None, step_show=1, use_pcolormesh=True):
        """
        Сохранить анимацию поля |U| для nsteps шагов по dt.
        Параметры:
            dt        - временной шаг, который будет использовать calc_dt
            nsteps    - число кадров/шагов интегрирования
            filename  - куда сохранять (mp4 или gif)
            interval  - миллисекунды между кадрами при просмотре (не влияет на файл)
            fps       - fps для сохранения (для mp4)
            cmap      - цветовая карта (по умолчанию 'plasma')
            vmin/vmax - если задать, цветовая шкала фиксируется; иначе берётся по данным первого кадра
            step_show - показывать каждую step_show точку в pcolormesh/quiver прореживании
            use_pcolormesh - если True, используем pcolormesh; иначе imshow
        Примечание: для сохранения mp4 нужен ffmpeg установленный в системе.
        """

        # --- подготовка фигуры/оси ---
        fig, ax = plt.subplots()

        # вычислим начальное поле U (на текущем self.zeta)
        # убедимся, что self.U актуально (если объект только создан, оно уже есть)
        U0 = self.U.copy()

        # фиксируем цветовую шкалу, если не задана
        if vmin is None:
            vmin = float(np.nanmin(U0))
        if vmax is None:
            vmax = float(np.nanmax(U0))

        # отрисовка начального кадра
        if use_pcolormesh:
            mesh = ax.pcolormesh(self.X, self.Y, U0, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # imshow удобен для регулярных сеток, но тогда нужно правильно выставить extent
            extent = [self.x[0], self.x[-1] + (self.x[1]-self.x[0]), self.y[0], self.y[-1] + (self.y[1]-self.y[0])]
            mesh = ax.imshow(U0, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(mesh, ax=ax, label="|U| (модуль скорости)")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Модуль скорости |U|")

        ax.set_aspect("equal")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        # --- функция обновления (каждый кадр делаем шаг интегрирования и обновляем данные) ---
        def update(frame):
            nonlocal mesh
            self.calc_dt(dt)

            if use_pcolormesh:
                mesh.remove()
                mesh = ax.pcolormesh(self.X, self.Y, self.U,
                                     shading="auto", cmap=cmap,
                                     vmin=vmin, vmax=vmax)
            else:
                mesh.set_data(self.U)

            ax.set_title(f"|U|, step {frame+1}/{nsteps}")
            print(f"step {frame+1}/{nsteps}")
            return mesh


        # --- создаём анимацию ---
        anim = FuncAnimation(fig, update, frames=nsteps, interval=interval, blit=False)

        # --- сохраняем ---
        # выбираем формат по расширению
        if filename.lower().endswith(".mp4"):
            # ffmpeg writer
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(filename, writer=writer, dpi=150)
            except Exception as e:
                plt.close(fig)
                raise RuntimeError("Не удалось сохранить mp4: проверьте, установлен ли ffmpeg. Ошибка: " + str(e))

        plt.close(fig)
        print(f"Анимация сохранена в {filename}")

    def animate_uv(self, dt, nsteps, filename="uv_anim.mp4", interval=50, fps=20,
                  cmap="viridis", vmin=None, vmax=None, step_show=1, use_pcolormesh=True):
        """
        Сохранить анимацию поля |U| для nsteps шагов по dt.
        Параметры:
            dt        - временной шаг, который будет использовать calc_dt
            nsteps    - число кадров/шагов интегрирования
            filename  - куда сохранять (mp4 или gif)
            interval  - миллисекунды между кадрами при просмотре (не влияет на файл)
            fps       - fps для сохранения (для mp4)
            cmap      - цветовая карта (по умолчанию 'plasma')
            vmin/vmax - если задать, цветовая шкала фиксируется; иначе берётся по данным первого кадра
            step_show - показывать каждую step_show точку в pcolormesh/quiver прореживании
            use_pcolormesh - если True, используем pcolormesh; иначе imshow
        Примечание: для сохранения mp4 нужен ffmpeg установленный в системе.
        """

        # --- подготовка фигуры/оси ---
        fig, ax = plt.subplots()

        # вычислим начальное поле U (на текущем self.zeta)
        # убедимся, что self.U актуально (если объект только создан, оно уже есть)
        U0 = self.U.copy()

        # фиксируем цветовую шкалу, если не задана
        if vmin is None:
            vmin = float(np.nanmin(U0))
        if vmax is None:
            vmax = float(np.nanmax(U0))

        # отрисовка начального кадра
        if use_pcolormesh:
            mesh = ax.pcolormesh(self.X, self.Y, U0, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # imshow удобен для регулярных сеток, но тогда нужно правильно выставить extent
            extent = [self.x[0], self.x[-1] + (self.x[1]-self.x[0]), self.y[0], self.y[-1] + (self.y[1]-self.y[0])]
            mesh = ax.imshow(U0, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(mesh, ax=ax, label="|U| (модуль скорости)")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Модуль скорости |U|")

        # --- стрелки: прореживание сетки и фиксированная длина в координатах ---
        # выберем прореживание для стрелок, чтобы не было их слишком много
        sx = step_show
        Xs = self.X[::sx, ::sx]
        Ys = self.Y[::sx, ::sx]
        us = self.u[::sx, ::sx]
        vs = self.v[::sx, ::sx]

        # нормируем и задаём длину стрелки в доле от расстояния между прореженными узлами
        mag = np.sqrt(us**2 + vs**2)
        mag[mag == 0] = 1.0
        usn = us / mag
        vsn = vs / mag

        dx_plot = (self.x[1] - self.x[0]) * sx
        dy_plot = (self.y[1] - self.y[0]) * sx
        arrow_length = min(dx_plot, dy_plot) * 0.5  # половина расстояния между прореженными точками

        U_plot = usn * arrow_length
        V_plot = vsn * arrow_length

        quiv = ax.quiver(Xs, Ys, U_plot, V_plot, color="white",
                         angles="xy", scale_units="xy", scale=1, pivot="mid", width=0.003)

        ax.set_aspect("equal")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        # --- функция обновления (каждый кадр делаем шаг интегрирования и обновляем данные) ---
        def update(frame):
            nonlocal mesh, quiv  # обязательно в начале

            # шаг интегрирования: обновит self.U, self.u, self.v, self.zeta и т.д.
            self.calc_dt(dt)

            # --- обновляем pcolormesh / imshow ---
            if use_pcolormesh:
                # удаляем старый QuadMesh корректно
                mesh.remove()
                # создаём новый с актуальными данными
                mesh = ax.pcolormesh(self.X, self.Y, self.U,
                                     shading="auto", cmap=cmap,
                                     vmin=vmin, vmax=vmax)
            else:
                # imshow: просто обновляем данные
                mesh.set_data(self.U)

            # --- обновляем quiver: удаляем старый и создаём новый ---
            quiv.remove()
            us = self.u[::sx, ::sx]
            vs = self.v[::sx, ::sx]
            mag = np.sqrt(us**2 + vs**2)
            mag[mag == 0] = 1.0
            usn = us / mag
            vsn = vs / mag
            U_plot = usn * arrow_length
            V_plot = vsn * arrow_length
            quiv = ax.quiver(Xs, Ys, U_plot, V_plot, color="white",
                             angles="xy", scale_units="xy", scale=1,
                             pivot="mid", width=0.003)

            ax.set_title(f"|U|, step {frame+1}/{nsteps}")
            print(f"step {frame+1}/{nsteps}")
            return mesh, quiv


        # --- создаём анимацию ---
        anim = FuncAnimation(fig, update, frames=nsteps, interval=interval, blit=False)

        # --- сохраняем ---
        # выбираем формат по расширению
        if filename.lower().endswith(".mp4"):
            # ffmpeg writer
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(filename, writer=writer, dpi=150)
            except Exception as e:
                plt.close(fig)
                raise RuntimeError("Не удалось сохранить mp4: проверьте, установлен ли ffmpeg. Ошибка: " + str(e))
        elif filename.lower().endswith(".gif"):
            # можно сохранить gif (требует ImageMagick или pillow)
            anim.save(filename, writer="pillow", fps=fps)
        else:
            # попытка автосохранения
            anim.save(filename, fps=fps)

        plt.close(fig)
        print(f"Анимация сохранена в {filename}")


@dataclass
class ConfigETDRK4:
    name: str = "zonostrophic"
    Ny: int = 256
    Nx: int = 256
    beta: float = 1.0
    nu: float = 1e-14          # коэффициент hyperviscosity
    mu: float = 0.01824        # линейная фрикция
    hyper_order: int = 4       # порядок hyperviscosity
    kf: float = 32.0           # центральная волновая длина форсирования
    dk: float = 1.0            # ширина кольца форсирования
    dt: float = 0.05
    forc_amp: float = 1e-6

def apply_dealiasing_in_spectrum(s_hat, mask):
    """Обнуляет спектр вне маски (маска булева по kx,ky)."""
    if mask is None:
        return s_hat
    out = s_hat.copy()
    out[~mask] = 0.0
    return out

class AtmosphereETDRK4:
    def __init__(self, cfg=None):
        if cfg is None:
            return None
        self.cfg = cfg
        self.Nx = cfg.Nx
        self.Ny = cfg.Ny
        self.size = (self.Nx, self.Ny)

        # координаты
        self.x = np.linspace(0, 2*PI, self.Nx, endpoint=False)
        self.y = np.linspace(0, 2*PI, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # спектр
        self.kx = 2*PI*np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = 2*PI*np.fft.fftfreq(self.Ny, d=self.dy)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing="xy")
        self.K2 = self.Kx**2 + self.Ky**2
        self.K = np.sqrt(self.K2)
        self.K2[0,0] = 1.0  # избегаем деления на 0

        # деалайзинг 2/3 rule
        kx_cut = 2/3 * np.max(np.abs(self.kx))
        ky_cut = 2/3 * np.max(np.abs(self.ky))
        self.dealias_mask = (np.abs(self.Kx) <= kx_cut) & (np.abs(self.Ky) <= ky_cut)

        # кольцевое форсирование
        self.force_mask = ((self.K >= (cfg.kf - cfg.dk)) & (self.K <= (cfg.kf + cfg.dk))).astype(float)

        # начальные поля
        self.zeta = np.zeros(self.size)
        self.psi, self.u, self.v, self.U = self.calc_psi(self.zeta)

        # линейный спектральный оператор L_lin
        self.L_lin = -cfg.mu - cfg.nu*(self.K**(2*cfg.hyper_order)) - 1j*cfg.beta*self.Kx/self.K2
        self.L_lin[0,0] = -cfg.mu  # избегаем деления на 0 для k=0

        # предвычисляем коэффициенты ETDRK4
        self.precompute_etdrk4(dt=cfg.dt)

        # амплитуда форсирования
        self.forc_amp = cfg.forc_amp

    def calc_psi(self, zeta):
        z_hat = np.fft.fft2(zeta)
        psi_hat = -z_hat / self.K2
        psi_hat[0,0] = 0.0
        psi = np.real(np.fft.ifft2(psi_hat))

        # скорости
        u = -np.real(np.fft.ifft2(1j*self.Ky*psi_hat))
        v = np.real(np.fft.ifft2(1j*self.Kx*psi_hat))
        U = np.sqrt(u**2 + v**2)
        return psi, u, v, U

    def calc_N(self, zeta):
        """Нелинейный член в спектре"""
        z_hat = np.fft.fft2(zeta)
        dzeta_dx = np.real(np.fft.ifft2(1j*self.Kx*z_hat))
        dzeta_dy = np.real(np.fft.ifft2(1j*self.Ky*z_hat))
        psi_hat = -z_hat/self.K2
        psi_hat[0,0] = 0.0
        u = -np.real(np.fft.ifft2(1j*self.Ky*psi_hat))
        v = np.real(np.fft.ifft2(1j*self.Kx*psi_hat))

        adv = u*dzeta_dx + v*dzeta_dy
        adv_hat = np.fft.fft2(adv)
        adv_hat = apply_dealiasing_in_spectrum(adv_hat, self.dealias_mask)

        # спектральное форсирование
        f_hat = self.calc_forcing()
        N_hat = -adv_hat + f_hat
        return N_hat

    def calc_forcing(self):
        re = rng.standard_normal(self.size)
        im = rng.standard_normal(self.size)
        f_hat = (re + 1j*im) * self.force_mask
        # Hermitian симметрия
        f_hat_shift = np.fft.fftshift(f_hat)
        f_hat_sym = 0.5 * (f_hat_shift + np.conj(np.flipud(np.fliplr(f_hat_shift))))
        f_hat = np.fft.ifftshift(f_hat_sym)
        f_hat *= self.forc_amp * np.sqrt(self.cfg.dt)
        f_hat = apply_dealiasing_in_spectrum(f_hat, self.dealias_mask)
        return f_hat

    def precompute_etdrk4(self, dt):
        L = self.L_lin
        self.dt = dt
        self.E = np.exp(dt*L)
        self.E2 = np.exp(dt*L/2.0)

        M = 16
        r = np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)
        LR = dt*L[:,:,None] + r[None,None,:]
        self.Q = dt*np.mean((np.exp(LR/2)-1)/LR, axis=2)
        self.f1 = dt*np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=2)
        self.f2 = dt*np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=2)
        self.f3 = dt*np.mean((-4-3*LR-LR**2 + np.exp(LR)*(4-LR))/LR**3, axis=2)

    def step(self):
        z_hat = np.fft.fft2(self.zeta)

        N1 = self.calc_N(self.zeta)
        a = self.E2 * z_hat + self.Q * N1
        N2 = self.calc_N(np.real(np.fft.ifft2(a)))
        b = self.E2 * z_hat + self.Q * N2
        N3 = self.calc_N(np.real(np.fft.ifft2(b)))
        c = self.E * z_hat + self.f1*N1 + 2*self.f2*(N2+N3)
        N4 = self.calc_N(np.real(np.fft.ifft2(c)))

        z_hat_new = self.E * z_hat + self.f1*N1 + 2*self.f2*(N2+N3) + self.f3*N4

        # обновляем поля
        self.zeta = np.real(np.fft.ifft2(z_hat_new))
        self.psi, self.u, self.v, self.U = self.calc_psi(self.zeta)

        # автоподгонка forc_amp
        self._autotune_forcing_stable()
        return self.zeta, self.psi, self.u, self.v, self.U

    def _autotune_forcing_stable(self, target_energy=1e-30, alpha=0.05, min_amp=1e-6, max_step=0.1):
        """
        Плавная автоподгонка forc_amp на основе текущей кинетической энергии.
        Вызывается в каждом шаге calc_dt.
        """
        # вычисляем текущую кинетическую энергию
        KE = 0.5 * np.mean(self.u**2 + self.v**2)
        KE = max(KE, 1e-30)  # предотвращаем деление на ноль

        # относительная корректировка
        factor = (target_energy / KE - 1.0) * alpha
        factor = np.clip(factor, -max_step, max_step)

        # обновляем forc_amp
        self.forc_amp *= (1.0 + factor)
        self.forc_amp = max(self.forc_amp, min_amp)

        # Опционально: лог для дебага
        print(f"KE={KE:.3e}, forc_amp={self.forc_amp:.3e}, factor={factor:.3e}")

    # --- статические графики ---
    def plot_x_grid(self):
        plt.figure()
        for xi in self.x:
            plt.plot([xi]*len(self.y), self.y, color="gray")
        for yi in self.y:
            plt.plot(self.x, [yi]*len(self.x), color="gray")
        plt.title("XY-grid")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.show()

    def plot_k_grid(self):
        plt.figure()
        for xi in self.kx:
            plt.plot([xi]*len(self.ky), self.ky, color="gray")
        for yi in self.ky:
            plt.plot(self.kx, [yi]*len(self.kx), color="gray")
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
        plt.show()

    def plot_uv(self, step=2, arrow_frac=0.5):
        plt.figure()
        pcm = plt.pcolormesh(self.X, self.Y, self.U, shading='auto', cmap='viridis')
        Xs, Ys = self.X[::step, ::step], self.Y[::step, ::step]
        us, vs = self.u[::step, ::step], self.v[::step, ::step]
        mag = np.sqrt(us**2 + vs**2)
        mag[mag==0]=1
        usn, vsn = us/mag, vs/mag
        dx_plot = (self.x[1]-self.x[0])*step
        dy_plot = (self.y[1]-self.y[0])*step
        arrow_length = min(dx_plot, dy_plot)*arrow_frac
        plt.quiver(Xs, Ys, usn*arrow_length, vsn*arrow_length,
                color='white', angles='xy', scale_units='xy', scale=1, pivot='mid', width=0.003)
        plt.colorbar(pcm, label='|U|')
        plt.title("Velocity field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.show()

    # --- анимации ---
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

        cbar = fig.colorbar(mesh, ax=ax, label=field)
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
            self.step()  # шаг интегрирования

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

        anim = FuncAnimation(fig, update, frames=nsteps, interval=interval, blit=False)

        if filename.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=fps)
            anim.save(filename, writer=writer, dpi=150)
        elif filename.lower().endswith(".gif"):
            anim.save(filename, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Анимация сохранена в {filename}")


config = Config()

atm = Atmosphere(config)

# atm.plot_zeta()
# atm.plot_U()
# atm.plot_uv(step=4)

#atm.animate_field(field='zeta', nsteps=5000, filename="zeta.mp4")
# atm.animate_field(field='U', nsteps=200, filename="U.mp4")

N = 500
for k in range(N):
    atm.step()
    print(f"zeta, step {k+1}/{N}")

atm.plot_zeta()
atm.plot_U()
atm.plot_uv()