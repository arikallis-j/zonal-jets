import time 
import jax.numpy as jnp
from jax import jit
from jax import lax
import numpy as np

# ----------------------------
# Параметры
# ----------------------------
N = 256             # число точек по каждой оси
L = 2 * jnp.pi      # размер домена (0, L)
nu = 1e-3           # вязкость
dt = 1e-3           # шаг по времени
steps = 2000        # число шагов
dealias = True

# ----------------------------
# Частоты для спектра
# ----------------------------
kx1d = jnp.fft.fftfreq(N, d=(L / N)) * 2 * jnp.pi  # частоты в физ. единицах
ky1d = kx1d
kx = kx1d[:, None]   # столбец
ky = ky1d[None, :]   # строка
k2 = kx**2 + ky**2
k2 = jnp.where(k2 == 0.0, 1.0, k2)  # чтобы не делить на ноль

# деалясинг: 2/3 rule mask
if dealias:
    kxmax = jnp.max(jnp.abs(kx1d))
    cutoff = 2.0/3.0 * kxmax
    KX, KY = jnp.meshgrid(jnp.abs(kx1d), jnp.abs(ky1d), indexing='ij')
    mask = jnp.where((KX < cutoff) & (KY < cutoff), 1.0, 0.0)
else:
    mask = jnp.ones((N, N))

# ----------------------------
# Вспомогательные спектральные операторы
# ----------------------------
ikx = 1j * kx
iky = 1j * ky

def fft2(x):
    return jnp.fft.fft2(x)

def ifft2(xhat):
    return jnp.fft.ifft2(xhat)

# ----------------------------
# Преобразования между вихрём и скоростью
# ----------------------------
def streamfunction_hat_from_omega_hat(omega_hat):
    # psi_hat = - omega_hat / k^2
    return -omega_hat / k2

def velocity_hat_from_streamfunction_hat(psi_hat):
    # u_x_hat = i ky * psi_hat  (note: derivative wrt y)
    ux_hat = iky * psi_hat
    uy_hat = -ikx * psi_hat
    return ux_hat, uy_hat

def velocity_from_omega_hat(omega_hat):
    psi_hat = streamfunction_hat_from_omega_hat(omega_hat)
    ux_hat, uy_hat = velocity_hat_from_streamfunction_hat(psi_hat)
    ux = jnp.real(ifft2(ux_hat))
    uy = jnp.real(ifft2(uy_hat))
    return ux, uy

# ----------------------------
# Правая часть для вихря dω/dt
# ----------------------------
@jit
def rhs(omega_hat):
    # деалясинг в спектре, чтобы исключить алиасинг при мультипликациях
    omega_hat_dealiased = omega_hat * mask

    # физ пространство
    omega = jnp.real(ifft2(omega_hat_dealiased))

    # скорость из вихря
    ux, uy = velocity_from_omega_hat(omega_hat_dealiased)

    # производные вихря в физ. пространстве
    domega_dx = jnp.real(ifft2(ikx * omega_hat_dealiased))
    domega_dy = jnp.real(ifft2(iky * omega_hat_dealiased))

    # нелинейное слагаемое u·grad ω (в физическом пространстве)
    adv = ux * domega_dx + uy * domega_dy

    # перевести адвекцию в спектр
    adv_hat = fft2(adv)

    # диффузионный член ν ∇^2 ω в спектре: -ν k^2 ω_hat
    diff_hat = -nu * k2 * omega_hat_dealiased

    # dω_hat/dt = - fft(u·∇ω) + diff_hat
    return -adv_hat + diff_hat

# ----------------------------
# RK4 шаг в спектре
# ----------------------------
@jit
def rk4_step(omega_hat, dt):
    k1 = rhs(omega_hat)
    k2 = rhs(omega_hat + 0.5 * dt * k1)
    k3 = rhs(omega_hat + 0.5 * dt * k2)
    k4 = rhs(omega_hat + dt * k3)
    return omega_hat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

@jit
def time_step(n, omega_hat):
    return rk4_step(omega_hat, dt)

@jit
def integrate(omega_hat0, steps):
    return lax.fori_loop(0, steps, time_step, omega_hat0)


def scan_step(omega_hat, _):
    omega_hat_next = rk4_step(omega_hat, dt)
    return omega_hat_next, omega_hat_next

@jit
def integrate_with_history(omega_hat0):
    omega_hat_final, omega_hat_all = lax.scan(scan_step, omega_hat0, None, length=steps)
    return omega_hat_final, omega_hat_all

# ----------------------------
# Начальное условие: Taylor-Green vortex (как пример)
# ----------------------------
x = jnp.linspace(0, L, N, endpoint=False)
y = jnp.linspace(0, L, N, endpoint=False)
X, Y = jnp.meshgrid(x, y, indexing='ij')

# классическое TG: u =  sin(x) cos(y), v = -cos(x) sin(y)  -> ω = -2 sin(x) sin(y)
omega0 = -2.0 * jnp.sin(X) * jnp.sin(Y)
omega0_hat = fft2(omega0)

# ----------------------------
# Основной цикл (JIT-компиляция шага; loop можно ускорить jax.lax.fori_loop)
# ----------------------------
omega_hat = omega0_hat


start = time.perf_counter()
omega_hat_final = integrate(omega0_hat, steps)
end = time.perf_counter()
print(end-start)

start = time.perf_counter()
omega_hat_final, omega_hat_all = integrate_with_history(omega0_hat)
end = time.perf_counter()
print(end-start)

# start = time.perf_counter()
# for n in range(steps):
#     omega_hat = rk4_step(omega_hat, dt)
# end = time.perf_counter()
# print(end-start)

# В конце — получить физич. поля
omega = jnp.real(ifft2(omega_hat))
ux, uy = velocity_from_omega_hat(omega_hat)

# Сохранить или визуализировать (например, с matplotlib, преобразовав к numpy)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.contourf(np.array(X), np.array(Y), np.array(omega), levels=60)
plt.colorbar()
plt.title("Vorticity")
plt.show()
plt.savefig('zeta.png')
