import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

N_ELEMENTS = 20

mesh = fe.UnitSquareMesh(N_ELEMENTS, N_ELEMENTS)

lagr_polinomial_first_order = fe.FunctionSpace(mesh,"Lagrange",2,)


u0 = fe.Expression("1 + 0.2*x[0]*x[0] + 0.1*x[1]*x[1]", degree = 1)
u0 = fe.interpolate(u0, lagr_polinomial_first_order)

#Эта функция нужна для скрытых методов сетки. on_boundary булева переменная,
#которая приходит из автоматической проверки точки на принедлежность границе.
#можно переписать функию и не использовать on_boundary.
def on_boundary(x, on_boundary):
    return on_boundary

#Граничные условия
bc = fe.DirichletBC(
    lagr_polinomial_first_order,
    u0,
    on_boundary,
)

#задаем функции для составления основного уравнения
u = fe.TrialFunction(lagr_polinomial_first_order)
v = fe.TestFunction(lagr_polinomial_first_order)
u_solution = fe.Function(lagr_polinomial_first_order)


#Определим f
sigma = 0.3
x0 = 0.2
y0 = 0.2

f = fe.Expression("7*exp(-0.5*(pow((x[0] - x0)/sigma, 2)) "\
                    " - 0.5*(pow((x[1] - y0)/sigma, 2)))",
                   x0=x0, y0=y0, sigma=sigma, degree = 2)

#Замечу, что константы определяются так, а не через expression. Можно попробовать подставить f1 в решение.
f1 = fe.Constant("0.0")


#составим уравнение
A = fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
L = f * v * fe.dx

fe.solve(A == L, u_solution, bc)

pl = fe.plot(u_solution)
#fe.plot(mesh) #сетка
plt.colorbar(pl)
plt.show()
#fe.interactive()

