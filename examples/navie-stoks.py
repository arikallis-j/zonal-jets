import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

N_POINTS_P_AXIS = 61

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 1000
KINEMATIC_VISCOSITY = 0.01 # -> Re = 100 Чудеса начинаются с Re = 2300
                            # Однако уже при Re = 1000 программа перестает работать


mesh = fe.UnitSquareMesh(N_POINTS_P_AXIS, N_POINTS_P_AXIS)

# Taylor-Hood Elements. Степень многочленов для давления должна быть на 1 меньше
#степени многочленов для скорости
velocity_function_space = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
pressure_function_space = fe.FunctionSpace(mesh, "Lagrange", 1)

u_trial = fe.TrialFunction(velocity_function_space)
p_trial = fe.TrialFunction(pressure_function_space)
v_test = fe.TestFunction(velocity_function_space)
q_test = fe.TestFunction(pressure_function_space)

# По бокам скорость равна 0. Сверху и снизу скорость равна 1. Граничные условия
stationary_wall_boundary_condition = fe.DirichletBC(
    velocity_function_space,
    (0.0, 0.0),
    """
    on_boundary && (x[0] < DOLFIN_EPS || x[0] > (1.0 - DOLFIN_EPS))
    """
)
moving_wall_boundary_condition = fe.DirichletBC(
    velocity_function_space,
    (1.0, 0.0),
    """
    on_boundary && (x[1] > (1.0 - DOLFIN_EPS) || x[1] < DOLFIN_EPS)
    """
)
velocity_boundary_conditions = [stationary_wall_boundary_condition, moving_wall_boundary_condition]


u_prev = fe.Function(velocity_function_space)
u_tent = fe.Function(velocity_function_space)
u_next = fe.Function(velocity_function_space)
p_next = fe.Function(pressure_function_space)

# Слабая форма для первого уравнения
momentum_weak_form_residuum = (
    1.0 / TIME_STEP_LENGTH * fe.inner(u_trial - u_prev, v_test) * fe.dx
    +
    fe.inner(fe.grad(u_prev) * u_prev, v_test) * fe.dx
    +
    KINEMATIC_VISCOSITY * fe.inner(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
)
momentum_weak_form_lhs = fe.lhs(momentum_weak_form_residuum)
momentum_weak_form_rhs = fe.rhs(momentum_weak_form_residuum)

# Слабая форма для второго уравнения на давление
pressure_poisson_weak_form_lhs = fe.inner(fe.grad(p_trial), fe.grad(q_test)) * fe.dx
pressure_poisson_weak_form_rhs = - 1.0 / TIME_STEP_LENGTH * fe.div(u_tent) * q_test * fe.dx

# Слабая форма для последнего уравнения. Переход к новому слою.
velocity_update_weak_form_lhs = fe.inner(u_trial, v_test) * fe.dx
velocity_update_weak_form_rhs = (
    fe.inner(u_tent, v_test) * fe.dx
    -
    TIME_STEP_LENGTH * fe.inner(fe.grad(p_next), v_test) * fe.dx
)

# Магия для ускорения вычисление.
momentum_assembled_system_matrix = fe.assemble(momentum_weak_form_lhs)
pressure_poisson_assembled_system_matrix = fe.assemble(pressure_poisson_weak_form_lhs)
velocity_update_assembled_system_matrix = fe.assemble(velocity_update_weak_form_lhs)

for t in tqdm(range(N_TIME_STEPS)):
    # (1) Решение первого уравнения
    momentum_assembled_rhs = fe.assemble(momentum_weak_form_rhs)
    [bc.apply(momentum_assembled_system_matrix, momentum_assembled_rhs) for bc in velocity_boundary_conditions]
    fe.solve(
        momentum_assembled_system_matrix,
        u_tent.vector(),
        momentum_assembled_rhs,
        "gmres",
        "ilu",
    )

    # (2) Решение второго
    pressure_poisson_assembled_rhs = fe.assemble(pressure_poisson_weak_form_rhs)
    fe.solve(
        pressure_poisson_assembled_system_matrix,
        p_next.vector(),
        pressure_poisson_assembled_rhs,
        "gmres",
        "amg",
    )

    # (3) Решение третьего
    [bc.apply(momentum_assembled_system_matrix, momentum_assembled_rhs) for bc in velocity_boundary_conditions]
    velocity_update_assembled_rhs = fe.assemble(velocity_update_weak_form_rhs)
    fe.solve(
        velocity_update_assembled_system_matrix,
        u_next.vector(),
        velocity_update_assembled_rhs,
        "gmres",
        "ilu",
    )
    u_prev.assign(u_next)

fe.plot(u_next)
plt.show()