"""
pseudospectral_vorticity_etdrk4.py

Псевдоспектральный решатель баротропного уравнения на b-плоскости
для относительной завихренности zeta_t + J(psi,zeta) + beta*v = j - m*zeta + nu_n Lap^{2n} zeta

Метод: ETDRK4 (Kassam & Trefethen style), 2/3 деалиасинг, ring forcing (white-in-time).
Автор: автоматически сгенерированная версия для воспроизведения рис.1 из
Srinivasan & Young (2012). Комментарии на русском.

Запуск:
    python pseudospectral_vorticity_etdrk4.py

Замечания:
 - Default N=256 (быстрее). Для высококачественной реплики статьи лучше N=512,
   но это значительно тяжелее по памяти/времени.
 - Параметр forc_amp управляет силой возмущения; чтобы точно соответствовать
   величине ε (в статье использовали скейлинг через ε) требуется тонкая настройка.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import stats

# ----------------------------
# Параметры роботы / физика
# ----------------------------
N = 32             # разрешение по x и y (измените на 512 для точной реплики статьи)
L = 2*np.pi         # физический домен: [0, 2πL] x [0, 2πL] при L=1 => длина 2π
# В статье используется домен 2πL x 2πL и kf*L = 32. Если L=1, то kf=32.
kf = 32.0           # forced wavenumber (kf*L = 32 в статье)
dk = 1.0            # ширина полосы forcing (будем делать кольцо толщины ~dk)
beta = 1.0          # можно масштабировать соответствие b*
m = 0.01824         # линейный drag (значение из рис.1)
n_hyper = 4         # порядок гипервязкости (в статье n=4)
nu_hyper = 1e-19    # коэффициент гипервязкости — подобрать под N и timestep (обычно мал)
forc_amp = 1.0e4    # амплитуда спектральной зашумленности (штука для настройки)
tmax = 200.0        # максимальное время интегрирования (в физических единицах)
dt = 0.01           # шаг времени (подбирать: слишком большой — неустойчивость)
plot_interval = 200 # через сколько шагов рисовать/сохранять (в шагах)
save_fields = True  # сохранять финальные поля (в памяти)

# ----------------------------
# Сетка и спектр
# ----------------------------
kx = np.fft.fftfreq(N, d=1.0/N) * N * (2*np.pi/L)  # спектральные волновые числа по x
ky = np.fft.fftfreq(N, d=1.0/N) * N * (2*np.pi/L)  # спектральные волновые числа по y
kx = kx.reshape(N,1)
ky = ky.reshape(1,N)
K2 = kx**2 + ky**2
K2[0,0] = 1.0  # избегаем деления на 0 (эпсилон-правка)
K = np.sqrt(K2)

# деалиасинг: 2/3 правило
kmax = np.max(np.abs(np.fft.fftfreq(N)*N))
dealias_cut = (np.abs(np.fft.fftfreq(N)*N) > (2.0/3.0*kmax))
dealias_mask_1d = ~dealias_cut
dealias_mask = np.outer(dealias_mask_1d, dealias_mask_1d).astype(float)

# окно forcing: кольцо вокруг kf с шириной dk
forcing_mask = ((K >= (kf-dk/2.0)) & (K <= (kf+dk/2.0))).astype(float)

# координаты физических полей (для вывода)
x = np.linspace(0, 2*np.pi*L, N, endpoint=False)
y = np.linspace(0, 2*np.pi*L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# ----------------------------
# ETDRK4 предвычисления
# ----------------------------
# Линейный оператор L в спектре действует так: L_hat = -m - nu * k^{2n} + i * beta * kx / k^2
# Причина: в RHS у нас -beta*v, а v = psi_x = i kx psi_hat, psi_hat = -zeta_hat / k^2.
# Подставляя: -beta*v_hat -> i*beta*kx*zeta_hat/k^2  (см подробности в рассуждении).
L_k = -m - nu_hyper * (K**(2*n_hyper)) + 1j * beta * kx / K2

# ETDRK4 коэффициенты (vectorized)
E = np.exp(L_k * dt)
E2 = np.exp(L_k * dt/2.0)
# используем формулы Kassam & Trefethen (стандарт)
def phi_coeffs(L, dt):
    a = L*dt
    small = np.abs(a) < 1e-6
    # для устойчивости: вычисляем коэффициенты через ряд Тейлора если a близко к 0
    # коэффициенты для ETDRK4 по Trefethen:
    # f1 = dt * (-4 - a + np.exp(a)*(4 - 3*a + a**2)) / a**3
    # f2 = dt * ( 2 + a + np.exp(a)*(-2 + a) ) / a**3
    # f3 = dt * (-4 - 3*a - a**2 + np.exp(a)*(4 - a) ) / a**3
    expa = np.exp(a)
    a2 = a*a
    a3 = a2*a
    f1 = dt * (-4.0 - a + expa*(4.0 - 3.0*a + a2)) / a3
    f2 = dt * (2.0 + a + expa*(-2.0 + a)) / a3
    f3 = dt * (-4.0 - 3.0*a - a2 + expa*(4.0 - a)) / a3
    # Taylor expansions for small a
    if np.any(small):
        # порядок 6 достаточен
        aa = a[small]
        f1[small] = dt*(1.0/6.0 - aa/24.0 + aa**2/120.0)
        f2[small] = dt*(1.0/2.0 - aa/6.0 + aa**2/24.0)
        f3[small] = dt*(1.0/3.0 - aa/8.0 + aa**2/30.0)
    return f1, f2, f3

f1, f2, f3 = phi_coeffs(L_k, dt)

# ----------------------------
# Начальные условия
# ----------------------------
# старт из покоя (нулевая завихренность) + малый рандом (опционально)
zeta = np.zeros((N,N), dtype=float)
zeta += 1e-6 * np.random.randn(N,N)   # малые шумы для инициации

# функция вычисления нелинейной части (в спектре)
def nonlinear_term(zeta_hat):
    # psi_hat = - zeta_hat / k^2
    psi_hat = - zeta_hat / K2
    psi_hat[0,0] = 0.0
    # velocity in physical space
    u = -np.real(ifft2(1j * ky * psi_hat))   # u = -psi_y ; note ky axis orientation
    v =  np.real(ifft2(1j * kx * psi_hat))   # v =  psi_x
    zeta_phys = np.real(ifft2(zeta_hat))
    # вычислим якобиан J = u * zeta_x + v * zeta_y = u zeta_x + v zeta_y
    zeta_x = np.real(ifft2(1j * kx * zeta_hat))
    zeta_y = np.real(ifft2(1j * ky * zeta_hat))
    J = u * zeta_x + v * zeta_y
    # возвращаем -FFT(J) (в RHS стоит -J)
    N_hat = -fft2(J)
    # деалиасинг
    N_hat *= dealias_mask
    return N_hat

# вспомогательная генерирующая функция белого по времени кольцевого forcing'а (в спектре)
def ring_forcing():
    # комплексные независимые гауссовы с нулевой ср. и var=1, но симметричны для реального поля
    xi = (np.random.randn(N,N) + 1j*np.random.randn(N,N)) * 0.5
    # применяем маску кольца
    Fk = xi * forcing_mask
    # делаем эрмитово-симметричным, чтобы поле реальное
    # (fft2 ожидает Hermitian symmetry for real physical)
    Fk = 0.5*(Fk + np.conj(np.flip(np.flip(Fk, axis=0), axis=1)))
    # умножаем на амплитуду и деалиасим
    Fk *= forc_amp
    Fk *= dealias_mask
    return Fk

# перевод начального поля в спектр
zeta_hat = fft2(zeta)

# ----------------------------
# Основной цикл интегрирования (ETDRK4)
# ----------------------------
nt = int(np.round(tmax / dt))
times = np.arange(0, nt) * dt

# для простого мониторинга сохраним U(y) и время
U_history = []
time_history = []
zeta_final = None

print("Start integration: N=%d, dt=%.4g, nt=%d" % (N, dt, nt))
for step in range(nt):
    t = step * dt

    # compute nonlinear term N1
    N1 = nonlinear_term(zeta_hat)

    # forcing: белый по времени -> в каждом шаге новая независимая реализация
    j_hat = ring_forcing()

    a = E2 * zeta_hat + (dt/2.0) * E2 * N1
    N2 = nonlinear_term(a)
    b = E2 * zeta_hat + (dt/2.0) * E2 * N2
    N3 = nonlinear_term(b)
    c = E * zeta_hat + dt * E * N3
    N4 = nonlinear_term(c)

    # ETDRK4 композиция (включая линейную часть через E и phi-коэффициенты)
    zeta_hat = E * zeta_hat + f1 * N1 + 2.0 * f2 * (N2 + N3) + f3 * N4

    # добавляем forcing (он аддитивен в RHS): используем простую ячейку forward Euler для белого шума
    # (т.е. j_hat * dt) — можно и в ETDRK4 включить, но для белого по времени часто достаточно такой вставки
    zeta_hat += dt * j_hat

    # (опционально) небольшая фильтрация/стабилизация - здесь не нужна из-за гипервязкости

    # мониторинг
    if (step % plot_interval) == 0 or step == nt-1:
        # вычисляем zonal mean velocity U(y) = average_x u(x,y)
        psi_hat = - zeta_hat / K2
        psi_hat[0,0] = 0.0
        u_phys = -np.real(ifft2(1j * ky * psi_hat))
        U = np.mean(u_phys, axis=0)  # усреднение по x -> функция y
        U_history.append(U.copy())
        time_history.append(t)
        print("step %d / %d, t=%.3f, max|zeta|=%.3e" % (step, nt, t, np.max(np.abs(np.real(ifft2(zeta_hat))))))

    # сохраняем финальное поле
    if step == nt-1:
        zeta_final = np.real(ifft2(zeta_hat))

# ----------------------------
# Визуализация результатов (примитивно)
# ----------------------------
if zeta_final is None:
    zeta_final = np.real(ifft2(zeta_hat))

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1)
# zonally averaged U(y)
U_final = np.mean(-np.real(ifft2(1j * ky * (-zeta_hat / K2))), axis=0)
ax1.plot(np.linspace(0, 2*np.pi*L, N, endpoint=False), U_final)
ax1.set_title("Zonally averaged U(y)")
ax1.set_xlabel("y")
ax1.set_ylabel("U")

ax2 = fig.add_subplot(1,2,2)
im = ax2.imshow(zeta_final.T, origin='lower',
                extent=[0, 2*np.pi*L, 0, 2*np.pi*L],
                aspect='auto')
ax2.set_title("zeta(x,y) final snapshot")
fig.colorbar(im, ax=ax2, orientation='vertical')
plt.tight_layout()
plt.show()

# ----------------------------
# Конец
# ----------------------------
print("Finished. Финальный шаг t=%.3f" % (nt*dt))
