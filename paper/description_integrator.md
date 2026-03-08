# Runge–Kutta methods

## Definition

$$\frac{dy_n}{dt_n} = f(t_n, y_n)$$

$$y_{n+1} = y_n + h \cdot \sum_{i = 1}^S b_i \cdot k_i$$

### RK1: S = 1

- $b_1 = 1, k_1 = f(t_n, y_n)$
$$y_n+1 = y_n + h \cdot k_1$$

### RK2: S = 2

- $b_1 = 1/2, k_1 = f(t_n, y_n)$
- $b_2 = 1/2, k_2 = f(t_n + h, y_n + h \cdot k_1)$
$$y_n+1 = y_n + \frac{h}{2} \cdot (k_1 + k_2)$$

### RK3: S = 3

- $b_1 = 1/6, k_1 = f(t_n, y_n)$
- $b_2 = 4/6, k_2 = f(t_n + h/2, y_n + h/2 \cdot k_1)$
- $b_3 = 1/6, k_3 = f(t_n + h, y_n - h \cdot k_1 + 2h \cdot k_2)$

$$y_{n+1} = y_n + \frac{h}{6} \cdot (k_1 + 4k_2 + k_3)$$

### RK4: S = 4

- $b_1 = 1/6, k_1 = f(t_n, y_n)$
- $b_2 = 2/6, k_2 = f(t_n + h/2, y_n + h/2 \cdot k_1)$
- $b_3 = 2/6, k_3 = f(t_n + h/2, y_n + h/2 \cdot k_2)$
- $b_4 = 1/6, k_4 = f(t_n + h, y_n + h \cdot k_3)$
$$y_{n+1} = y_n + \frac{h}{6} \cdot (k_1 + 2 k_2 + 2 k_3 + k_4)$$
