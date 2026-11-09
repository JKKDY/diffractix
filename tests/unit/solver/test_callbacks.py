"""Full numerical callbacks compose optical propagation and graph evaluation."""

import numpy as np
import pytest

from diffractix import Constraint, GaussianBeam, Objective, Parameter, Ray, Solver, Space, System
from diffractix.backend import jit, xp
from diffractix.graph import optimize


def build_callbacks(solver):
    objective_program, objective_descriptors, callable_objectives = solver._finalize_objectives()
    constraint_program, constraint_descriptors, callable_constraints = solver._finalize_constraints()
    return solver._build_callbacks(
        objective_program=optimize(objective_program),
        objective_descriptors=objective_descriptors,
        callable_objectives=callable_objectives,
        constraint_program=optimize(constraint_program),
        constraint_descriptors=constraint_descriptors,
        callable_constraints=callable_constraints,
        initial_values=solver.simulation.initial_values.values,
        use_jit=False,
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_full_callbacks_match_analytical_values_and_derivatives(use_jit):
    position, angle = Parameter(0.4).variable(), Parameter(0.3).variable()
    solver = Solver(System(Space(d=1), optical_input=Ray(x=position, theta=angle, n=1)))
    solver.target(Objective(position ** 2, weight=3), Objective(lambda result: result.final.x - 1, weight=2), lambda result: 7)
    solver.require(Constraint(position * angle), Constraint(lambda result: result.final.vector), Constraint(position * angle))
    callbacks, graph_rows, callable_rows = build_callbacks(solver)
    assert graph_rows == (range(0, 1), range(1, 3), range(3, 4))
    assert callable_rows == ()
    indices = solver.simulation.optics_program.parameter_indices
    p, a = indices[id(position)], indices[id(angle)]
    theta = solver.simulation.initial_values.values
    gradient = np.empty(2)
    gradient[p], gradient[a] = -0.432, -1.2
    objective_hessian = np.empty((2, 2))
    objective_hessian[p, p], objective_hessian[a, a] = 9.76, 4
    objective_hessian[p, a] = objective_hessian[a, p] = 4
    jacobian = np.empty((4, 2))
    jacobian[:, p], jacobian[:, a] = [0.3, 1, 0, 0.3], [0.4, 1, 1, 0.4]
    expected = {
        "objective": (49.2568, (theta,)),
        "gradient": (gradient, (theta,)),
        "objective_hessian": (objective_hessian, (theta,)),
        "constraints": ([0.12, 0.7, 0.3, 0.12], (theta,)),
        "jacobian": (jacobian, (theta,)),
        "constraint_hessian": ([[0, 5], [5, 0]], (theta, xp.array([1., 2., 3., 4.]))),
    }
    for name, (value, arguments) in expected.items():
        function = getattr(callbacks, name)
        if use_jit:
            function = jit(function)
        np.testing.assert_allclose(function(*arguments), value, atol=1e-12)


def test_all_result_symbol_kinds_and_lazy_result_sharing(monkeypatch):
    position = Parameter(0.4).variable()
    space = Space(d=1)
    simulation = System(space, optical_input=Ray(x=position, theta=0.3, n=1)).build()
    solver = Solver(simulation)
    solver.target(lambda result: result.initial.x + result.final.x)
    solver.require(*(Constraint(function) for function in (
        lambda result: result.initial.x, lambda result: result.final.x,
        lambda result: result[space].before.x, lambda result: result[space].after.x,
        lambda result: result[space].z_start, lambda result: result[space].z_end,
    )))
    calls = []
    run = simulation.run

    def record_run(theta):
        calls.append(1)
        return run(theta)

    monkeypatch.setattr(simulation, "run", record_run)
    callbacks, _, _ = build_callbacks(solver)
    calls.clear()
    np.testing.assert_allclose(callbacks.constraints(xp.array([0.4])), [0.4, 0.7, 0.4, 0.7, 0, 1])
    assert len(calls) == 1
    assert callbacks.objective(xp.array([0.4])) == pytest.approx(1.21)
    assert len(calls) == 2
    np.testing.assert_allclose(callbacks.constraints(xp.array([0.7])), [0.7, 1, 0.7, 1, 0, 1])
    assert len(calls) == 3


def test_empty_constraints_have_correct_derivative_shapes_and_direct_nodes_do_not_run_simulation(monkeypatch):
    parameter = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.target(parameter - 1)
    monkeypatch.setattr(solver.simulation, "run", lambda theta: pytest.fail("Direct nodes need no optical propagation"))
    callbacks, graph_rows, callable_rows = build_callbacks(solver)
    assert graph_rows == callable_rows == ()
    assert callbacks.objective(xp.array([2.])) == pytest.approx(1)
    np.testing.assert_allclose(callbacks.gradient(xp.array([2.])), [2])
    assert callbacks.constraints(xp.array([2.])).shape == (0,)
    assert callbacks.jacobian(xp.array([2.])).shape == (0, 1)
    np.testing.assert_array_equal(callbacks.constraint_hessian(xp.array([2.]), xp.zeros(0)), [[0]])


def test_gaussian_runtime_symbols_preserve_complex_propagation_derivatives():
    distance = Parameter(0.5).variable()
    wavelength = 1e-6
    solver = Solver(System(Space(d=distance), optical_input=GaussianBeam(q=1j, wavelength=wavelength, n=1)))
    solver.target(lambda result: result.final.w)
    solver.require(Constraint(lambda result: result.final.w ** 2))
    callbacks, _, _ = build_callbacks(solver)
    theta = xp.array([0.5])
    # q = d + i gives w**2 = wavelength * (1 + d**2) / pi.
    scale = wavelength / np.pi
    assert callbacks.objective(theta) == pytest.approx(1.25 * scale)
    np.testing.assert_allclose(callbacks.gradient(theta), [scale])
    np.testing.assert_allclose(callbacks.objective_hessian(theta), [[2 * scale]])
    # The full constraint vector also includes Space's nonnegative length.
    np.testing.assert_allclose(callbacks.constraints(theta), [0.5, 1.25 * scale])
    np.testing.assert_allclose(callbacks.jacobian(theta), [[1], [scale]])
    np.testing.assert_allclose(jit(callbacks.constraint_hessian)(theta, xp.array([0., 3.])), [[6 * scale]])
