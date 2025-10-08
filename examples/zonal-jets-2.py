#!/usr/bin/env python3
"""
zonostrophic_etdrk4_tuned.py

Псевдоспектральное решение баротропного уравнения вихря на β-плоскости.
Реализует ETDRK4 (Kassam-Trefethen), деалясинг 2/3, гипервязкость, белое
по времени спектральное форсирование (кольцо), и автоподгонку амплитуды
forc_amp. Сохраняет Hovmöller и финальные изображения в output/.
"""
import os
import time
import numpy as np
import matplotlib
# если запускаешь в headless-окружении, безопасно использовать Agg
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

# ----------------------- ПАРАМЕТРЫ СИМУЛЯЦИИ -----------------------
Nx = Ny = 256          # разрешение (можно уменьшить до 128 для быстрого теста)
Lx = Ly = 2*np.pi
dx = Lx / Nx
dy = Ly / Ny

# физические параметры
beta = 1.0
mu = 0.01824
nu = 1e-14
hyper_n = 4

# forcing (кольцо)
kf = 32.0
dk = 1.0
# начальная амплитуда — будет откалибрована autotune
forc_amp = 1e-6

# временные параметры (могут быть автоматически уменьшены при крупной подгонке)
dt = 0.02
nsteps = 5000
save_every = 10     # сохранять U(y) каждые save_every шагов
print_every = 100   # печатать диагностику каждые print_every шагов

# autotune параметры
Fr_target = 1e-3    # целевой rms форсинга в физпространстве (пример: 1e-3)
max_mul = 1e4       # максимальный множитель амплитуды за одну autotune итерацию


Fr_target = 3e-3
mu = 0.015
nu = 5e-15
dt = 0.01   # или 0.005 для устойчивости
nsteps = 10000


# output
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# RNG
rng = np.random.default_rng(0)

# ----------------------- ПОДГОТОВКА СПЕКТРАЛЬНЫХ МАТРИЦ -----------------------
kx = 2*np.pi * fftfreq(Nx, d=dx)
ky = 2*np.pi * fftfreq(Ny, d=dy)
kxx, kyy = np.meshgrid(kx, ky, indexing='xy')
k2 = kxx**2 + kyy**2
k2[0,0] = 1.0  # защита от деления на ноль (сразу зануляем среднюю компоненту позже)

# деалясинг 2/3
kx_max = np.max(np.abs(kx))
ky_max = np.max(np.abs(ky))
kx_cut = (2.0/3.0) * kx_max
ky_cut = (2.0/3.0) * ky_max
dealias_mask = (np.abs(kxx) <= kx_cut) & (np.abs(kyy) <= ky_cut)

# маска форсирования (кольцо)
kh = np.sqrt(kxx**2 + kyy**2)
force_mask = ((kh >= (kf - dk)) & (kh <= (kf + dk))).astype(float)

# линейный оператор спектрально: L(k) = -mu - nu*k^{2n} + i*beta*kx/k^2
k2n = (k2 ** hyper_n)
# L_lin будет создаваться/пересчитываться при необходимости dt менять не надо включать dt в L
def build_L_lin(mu_local, nu_local):
    Lvisc = - mu_local - (nu_local * k2n)
    Llin = Lvisc + 1j * beta * (kxx / k2)
    Llin[0,0] = - mu_local
    return Llin

L_lin = build_L_lin(mu, nu)

# ----------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------
def dealias(spec):
    return spec * dealias_mask

def invert_poisson_hat(zeta_hat):
    psi_hat = - zeta_hat / k2
    psi_hat[0,0] = 0.0
    return psi_hat

def forcing_hat_white_once(dt_local, amplitude):
    """Сгенерировать спектральный белый шум на шаг dt (один realization).
       Возвращает комплексный массив в спектре, с hermitian-симметрией."""
    re = rng.standard_normal((Ny, Nx))
    im = rng.standard_normal((Ny, Nx))
    fh = (re + 1j*im) * force_mask
    # симметризация (Hermitian) для реального поля
    fh_shift = fftshift(fh)
    fh_sym = 0.5 * (fh_shift + np.conj(np.flipud(np.fliplr(fh_shift))))
    fh = ifftshift(fh_sym)
    # масштаб: белый шум ~ sqrt(dt) и амплитуда amplitude
    fh *= amplitude * np.sqrt(dt_local)
    fh = dealias(fh)
    return fh

def forcing_rms_from_hat(fh):
    f_phys = np.real(ifft2(fh))
    return np.sqrt(np.mean(f_phys**2))

def compute_forcing_rms(amplitude, dt_local):
    fh = forcing_hat_white_once(dt_local, amplitude)
    return forcing_rms_from_hat(fh)

def autotune_forc_amp(forc_amp_local, dt_local, Fr_target_local=Fr_target, max_mul_local=max_mul):
    Fr_cur = compute_forcing_rms(forc_amp_local, dt_local)
    if Fr_cur <= 0:
        mul = max_mul_local
    else:
        mul = float(Fr_target_local / Fr_cur)
    if mul <= 1.0:
        print(f"[autotune] forcing_rms already >= target ({Fr_cur:.3e} >= {Fr_target_local:.3e}); no change")
        return forc_amp_local, Fr_cur, 1.0
    mul = min(mul, max_mul_local)
    new_amp = forc_amp_local * mul
    print(f"[autotune] Fr_cur={Fr_cur:.3e} -> target={Fr_target_local:.3e}, mul={mul:.3g}, new_amp={new_amp:.3e}")
    return new_amp, Fr_cur, mul

def etdrk4_coeffs(L, dt_local, M=32):
    """Compute ETDRK4 coefficients per Kassam & Trefethen."""
    E = np.exp(L * dt_local)
    E2 = np.exp(L * dt_local / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    L_flat = L.flatten()
    LR = dt_local * L_flat[:, None] + r[None, :]
    # avoid division by zero by relying on numpy complex arithmetic
    Q = dt_local * np.mean( (np.exp(LR/2.0) - 1.0) / LR , axis=1 ).reshape(L.shape)
    f1 = dt_local * np.mean( (-4.0 - LR + np.exp(LR) * (4.0 - 3.0*LR + LR**2)) / (LR**3), axis=1 ).reshape(L.shape)
    f2 = dt_local * np.mean( (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1 ).reshape(L.shape)
    f3 = dt_local * np.mean( (-4.0 - 3.0*LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=1 ).reshape(L.shape)
    return E, E2, Q, f1, f2, f3

def compute_nonlinear_hat(zeta_hat):
    """Compute -J_hat (spectral) where J = u*dzeta/dx + v*dzeta/dy (conservative with our sign conv.)."""
    psi_hat = invert_poisson_hat(zeta_hat)
    # velocities in physical space
    u = - np.real(ifft2(1j * kyy * psi_hat))  # u = -psi_y
    v =   np.real(ifft2(1j * kxx * psi_hat))  # v =  psi_x
    zeta = np.real(ifft2(zeta_hat))
    zeta_x = np.real(ifft2(1j * kxx * zeta_hat))
    zeta_y = np.real(ifft2(1j * kyy * zeta_hat))
    J = u * zeta_x + v * zeta_y
    J_hat = fft2(J)
    J_hat = dealias(J_hat)
    return -J_hat

def kinetic_energy_from_zeta_hat(zeta_hat):
    psi_hat = invert_poisson_hat(zeta_hat)
    u_hat = -1j * kyy * psi_hat
    v_hat =  1j * kxx * psi_hat
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))
    KE = 0.5 * np.mean(u*u + v*v)
    return KE, u, v

# ----------------------- ИНИЦИАЛИЗАЦИЯ -----------------------
zeta0 = 1e-6 * rng.standard_normal((Ny, Nx))
zeta_hat = fft2(zeta0)
zeta_hat = dealias(zeta_hat)

# autotune forc_amp (безопасно, с уменьшением dt при больших множителях)
forc_amp_cur = forc_amp
new_amp, Fr_cur, mul = autotune_forc_amp(forc_amp_cur, dt, Fr_target, max_mul)
# если mul большой (>100), уменьшаем dt вдвое и пересчитываем ETDRK4 коэфф-ты; повторить до тех пор, пока mul<=100 или dt очень мал
if mul > 100:
    print("[autotune] large multiplicator detected -> reducing dt by 2 to be conservative.")
    dt = max(dt/2.0, 1e-4)
    print(f"[autotune] new dt = {dt}")
    # повторно подгоним (ограничим второй раз)
    new_amp2, Fr_cur2, mul2 = autotune_forc_amp(new_amp, dt, Fr_target, max_mul)
    # если mul2 > max_mul, ограничим
    forc_amp_cur = new_amp2
else:
    forc_amp_cur = new_amp

# финальные ETDRK4 коэффициенты (с текущим dt)
L_lin = build_L_lin(mu, nu)
E, E2, Q_coefs, f1_coefs, f2_coefs, f3_coefs = etdrk4_coeffs(L_lin, dt, M=32)
print(f"Starting simulation with Nx={Nx}, dt={dt}, forc_amp={forc_amp_cur:.3e}, mu={mu}, nu={nu}")

# ----------------------- КОРРЕКТНЫЙ ВРЕМЕННОЙ ЦИКЛ (ETDRK4) -----------------------
U_hist = []
time_hist = []
t = 0.0
t0 = time.time()

# см. рекомендация: здесь мы генерируем ровно один forcing на шаг и используем его для всех стадий
for istep in range(1, nsteps+1):
    # сгенерировать forcing один раз на шаг
    F_step = forcing_hat_white_once(dt, forc_amp_cur)

    # N1
    N1 = compute_nonlinear_hat(zeta_hat) + F_step
    a = E2 * zeta_hat + Q_coefs * N1

    Na = compute_nonlinear_hat(a) + F_step
    b = E2 * zeta_hat + Q_coefs * Na

    Nb = compute_nonlinear_hat(b) + F_step
    c = E * zeta_hat + Q_coefs * (2.0 * Nb)

    Nc = compute_nonlinear_hat(c) + F_step

    # ETDRK4 update
    zeta_hat = E * zeta_hat + f1_coefs * N1 + 2.0 * f2_coefs * (Na + Nb) + f3_coefs * Nc
    zeta_hat = dealias(zeta_hat)

    t += dt

    # сохранить U по save_every
    if istep % save_every == 0 or istep == 1:
        KE, u_field, v_field = kinetic_energy_from_zeta_hat(zeta_hat)
        psi_hat_tmp = invert_poisson_hat(zeta_hat)
        u_phys = - np.real(ifft2(1j * kyy * psi_hat_tmp))
        U = np.mean(u_phys, axis=1)
        U_hist.append(U.copy())
        time_hist.append(t)

    # печать диагностики
    if istep % print_every == 0 or istep == 1:
        # вычислим forcing_rms в физ.пространстве (на текущем шаге)
        Fr_now = forcing_rms_from_hat(F_step)
        KE, _, _ = kinetic_energy_from_zeta_hat(zeta_hat)
        zeta_phys = np.real(ifft2(zeta_hat))
        print(f"step {istep}/{nsteps}, t={t:.3f}, Fr={Fr_now:.3e}, KE={KE:.3e}, max|zeta|={np.max(np.abs(zeta_phys)):.3e}")

# ----------------------- ПОСТ-ОБРАБОТКА -----------------------
zeta = np.real(ifft2(zeta_hat))
U_final = np.mean(- np.real(ifft2(1j * kyy * invert_poisson_hat(zeta_hat))), axis=1)
U_array = np.array(U_hist)  # shape (nt, Ny) with save_every steps -> transpose below
hov = U_array.T if U_array.size else np.zeros((Ny,1))

# Сохраним диагностику и массивы
np.savez(os.path.join(outdir, "result.npz"),
         zeta=zeta, U=U_final, U_hist=U_array, times=np.array(time_hist),
         params=dict(Nx=Nx, Ny=Ny, dt=dt, forc_amp=forc_amp_cur, mu=mu, nu=nu, kf=kf))

# Визуализация: сохраняем финальные графики
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(2,2,1)
im1 = ax1.imshow(zeta, origin='lower', extent=[0,Lx,0,Ly], cmap='RdBu_r')
ax1.set_title("Final vorticity ζ(x,y)")
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(2,2,2)
y_vals = np.linspace(0, Ly, Ny)
ax2.plot(y_vals, U_final)
ax2.set_title("Final zonal mean U(y)")
ax2.set_xlabel("y")

ax3 = fig.add_subplot(2,1,2)
if U_array.size:
    t_vals = np.array(time_hist)
    extent = [t_vals[0], t_vals[-1], 0, Ly]
    im3 = ax3.imshow(hov, origin='lower', aspect='auto', extent=extent, cmap='RdBu_r')
    ax3.set_title("Hovmöller: U(y,t)")
    ax3.set_xlabel("t")
    ax3.set_ylabel("y")
    plt.colorbar(im3, ax=ax3, fraction=0.04)
else:
    ax3.text(0.5,0.5,"No U history saved", ha='center')

plt.tight_layout()
pngfile = os.path.join(outdir, "final_figure.png")
fig.savefig(pngfile, dpi=200)
plt.close(fig)
print("Saved final figure to", pngfile)
print("Saved data to", os.path.join(outdir, "result.npz"))
print("Total runtime: {:.1f} s".format(time.time() - t0))
