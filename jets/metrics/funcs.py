import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model_saturation(t, alpha, A, t_sat):
    """
        { A * (t/t_sat)^alpha , t<t_sat
    y = {
        { A, t>=t_sat
    """
    t = np.asarray(t)
    if np.any(t <= 0):
        raise ValueError("All t must be positive")
    log_t = np.log(t)
    log_t_sat = np.log(t_sat)
    log_A = np.log(A)
    log_y = np.where(t < t_sat, alpha * (log_t - log_t_sat) + log_A, log_A)
    return np.exp(log_y)


def fit_saturation(T, Y, min_points=1, use_optimization=True):
    T = np.asarray(T)
    Y = np.asarray(Y)
    if len(T) != len(Y):
        raise ValueError("T and Y must have the same length")
    if np.any(T <= 0) or np.any(Y <= 0):
        raise ValueError("T and Y must be positive")
    if len(T) < 2 * min_points:
        raise ValueError(f"Not enough points with min_points={min_points}")

    # Сортировка
    idx = np.argsort(T)
    T = T[idx]
    Y = Y[idx]
    logT = np.log(T)
    logY = np.log(Y)
    n = len(T)

    # ---- 1. Перебор по сетке (дискретный поиск) ----
    best_rss = np.inf
    best_a = best_b = best_t_sat = None

    for i in range(min_points, n):
        t_sat_candidate = T[i]
        log_t_sat = logT[i]

        x = np.concatenate([logT[:i], np.full(n - i, log_t_sat)])
        X = np.column_stack((x, np.ones(n)))
        a, b = np.linalg.lstsq(X, logY, rcond=None)[0]
        pred = a * x + b
        rss = np.sum((logY - pred) ** 2)

        if rss < best_rss:
            best_rss = rss
            best_a, best_b, best_t_sat = a, b, t_sat_candidate

    if not use_optimization:
        c = best_a * np.log(best_t_sat) + best_b
        a, b, t_sat = best_a, best_b, best_t_sat
        return {'alpha': a, 'A': np.exp(c), 't_sat': t_sat}

    # ---- 2. Непрерывная оптимизация (уточнение t_sat) ----
    def rss_for_params(params):
        a, b, t_sat = params
        if t_sat <= T[0] or t_sat >= T[-1]:
            return 1e10  # штраф
        log_t_sat = np.log(t_sat)
        log_y_pred = np.where(T < t_sat, a * logT + b, a * log_t_sat + b)
        return np.sum((logY - log_y_pred) ** 2)

    # Начальное приближение из дискретного поиска
    x0 = [best_a, best_b, best_t_sat]
    # Ограничения: t_sat в пределах данных (можно также задать bounds)
    bounds = [(None, None), (None, None), (T[0], T[-1])]
    result = minimize(rss_for_params, x0, bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print("Оптимизация не сошлась, возвращаю дискретное решение.")
        a, b, t_sat = best_a, best_b, best_t_sat
        rss = best_rss
    else:
        a, b, t_sat = result.x
        rss = result.fun

    c = a * np.log(t_sat) + b
    return {'alpha': a, 'A': np.exp(c), 't_sat': t_sat}

def find_saturation(t, y, crit=3):
    found = False
    params_est = fit_saturation(t, y, use_optimization=False)
    params_true = fit_saturation(t, y, use_optimization=True)
    idx = np.searchsorted(t, params_est['t_sat'])
    inv_idx = len(t)-1-idx
    if inv_idx >= crit:
        found = True
    return params_true, found

def plot_saturation(t, y, params, title):
    T = np.linspace(0.5, 2048, 10000)
    Y = model_saturation(T, alpha=params['alpha'], A=params['A'], t_sat=params['t_sat'])
    plt.figure(figsize=(8, 5))
    plt.scatter(t, y, label='data',  color='black')
    plt.plot(T, Y , label='fit')
    plt.axvline(params['t_sat'], color='r', linestyle='--', label='t_sat')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
    print(f"{title}:")
    print(f"A = {params['A'].astype(float):.2e}")
    print(f"alpha = {params['alpha'].astype(float):.2f}")
    print(f"t_sat = {params['t_sat'].astype(float):.0f}")