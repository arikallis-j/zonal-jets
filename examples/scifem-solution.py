import numpy as np
from skfem import *
from skfem.helpers import grad, div, dot
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

from skfem.helpers import grad, div, dot


# ------------------------------
# 1. Сетка: прямоугольник 0..1 x 0..1, 10x10 делений
# ------------------------------
mesh = MeshTri.init_tensor(np.linspace(0, 1, 11), np.linspace(0, 1, 11))

# ------------------------------
# 2. Базисы
# ------------------------------
V = Basis(mesh, ElementTriP2())  # скорость P2
Q = Basis(mesh, ElementTriP1())  # давление P1

# ------------------------------
# 3. Параметры
# ------------------------------
nu = 0.01
f = np.array([0.0, 0.0])

# ------------------------------
# 4. Слабые формы
# ------------------------------

@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v)) * np.array(nu) * w


@BilinearForm
def b(u, p, w):
    return -div(u) * p * w

@LinearForm
def l(v, w):
    return dot(f, v) * w

# ------------------------------
# 5. Сборка матриц
# ------------------------------
A = a.assemble(V)
B = b.assemble(V, Q)
rhs = l.assemble(V)

# ------------------------------
# 6. Составляем блоковую систему
# ------------------------------
Z = csr_matrix((Q.N, Q.N))  # нулевая матрица для давления
M = bmat([[A, B.T],
          [B, Z]], format='csr')

rhs_full = np.concatenate([rhs, np.zeros(Q.N)])

# ------------------------------
# 7. Граничные условия (скорость=0 на границе)
# ------------------------------
boundary = mesh.boundary_nodes()
M = M.tolil()
rhs_full[boundary] = 0
for i in boundary:
    M[i, :] = 0
    M[i, i] = 1
M = M.tocsr()

# ------------------------------
# 8. Решение
# ------------------------------
sol = spsolve(M, rhs_full)

# ------------------------------
# 9. Разделяем скорость и давление
# ------------------------------
u_sol = sol[:V.N]
p_sol = sol[V.N:]

print("Скорость:", u_sol)
print("Давление:", p_sol)
