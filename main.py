import atmozon as atz
import time 
import jax

N = 256
steps = 1000 

atm = atz.Atmosphere()
grid = atz.Grid(N = 256)

start = time.perf_counter()
grid.calc(atm, steps=steps)
end = time.perf_counter()
print(end-start)

start = time.perf_counter()
grid.plot_U()
grid.plot_zeta()
end = time.perf_counter()
print(end-start)