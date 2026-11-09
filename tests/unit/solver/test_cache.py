"""Input validation and fixed Simulation ownership for reusable Solver callbacks."""

from unittest.mock import patch

import numpy as np
import pytest

from diffractix import Constraint, Parameter, Ray, Solver, System
from diffractix.backend import current_backend, xp
from diffractix.graph import Literal
from diffractix.solver import solver as solver_module
from tests.helpers.backend import capture_problem


@pytest.fixture
def solver():
    parameter = Parameter(1.5).variable(bounds=(1.0, 4.0))
    return Solver(System(optical_input=Ray(x=parameter, theta=0, n=1))).target(parameter - 0.2)


@pytest.mark.parametrize("argument", ["x0", "x_lower", "x_upper"])
@pytest.mark.parametrize("value", [1.5, [[1.5]], [1.0, 2.0]])
def test_override_arrays_require_canonical_vector_shape(solver, argument, value):
    with patch.object(solver, "_build_cache") as build, patch.object(
        solver, "_update_reduction",
    ) as reduce, patch.object(solver_module, "solve_scipy") as dispatch:
        with pytest.raises(ValueError, match=rf"{argument}.*shape"):
            solver.solve(**{argument: value})
    build.assert_not_called()
    reduce.assert_not_called()
    dispatch.assert_not_called()
    assert solver._cache is None


@pytest.mark.parametrize("overrides, message", [
    ({"x0": [np.nan]}, "x0.*finite"),
    ({"x0": [np.inf]}, "x0.*finite"),
    ({"x0": [-np.inf]}, "x0.*finite"),
    ({"x_lower": [np.nan]}, "bounds.*NaN"),
    ({"x_upper": [np.nan]}, "bounds.*NaN"),
    ({"x_lower": [3.0], "x_upper": [2.0]}, "x_lower.*x_upper"),
    ({"x0": [0.5]}, "x0.*within"),
    ({"x0": [4.5]}, "x0.*within"),
    ({"x_lower": [2.0]}, "x0.*within"),
    ({"x_upper": [1.0]}, "x0.*within"),
])
def test_invalid_overrides_fail_before_compilation_reduction_or_dispatch(solver, overrides, message):
    with patch.object(solver, "_build_cache") as build, patch.object(
        solver, "_update_reduction",
    ) as reduce, patch.object(solver_module, "solve_scipy") as dispatch:
        with pytest.raises(ValueError, match=message):
            solver.solve(**overrides)
    build.assert_not_called()
    reduce.assert_not_called()
    dispatch.assert_not_called()
    assert solver._cache is None


@pytest.mark.parametrize("start, lower, upper", [
    (1.5, -np.inf, np.inf),
    (1.5, -np.inf, 4.0),
    (1.5, 1.0, np.inf),
    (1.0, 1.0, 4.0),
    (4.0, 1.0, 4.0),
    (1.5, 1.5, 1.5),
])
def test_overrides_accept_infinite_and_inclusive_bounds(solver, start, lower, upper):
    problem = capture_problem(solver, x0=[start], x_lower=[lower], x_upper=[upper])
    np.testing.assert_array_equal(problem.x0, [start])
    np.testing.assert_array_equal(problem.x_lower, [lower])
    np.testing.assert_array_equal(problem.x_upper, [upper])
    np.testing.assert_array_equal(solver.simulation.initial_values.values, [1.5])
    assert [(p.lower_bound, p.upper_bound) for p in solver.simulation.parameter_data] == [(1., 4.)]


def test_invalid_later_solve_preserves_cache_and_next_solve_defaults(solver):
    capture_problem(solver)
    cache = solver._cache
    with pytest.raises(ValueError, match="x0.*within"):
        solver.solve(x_lower=[2.0])
    assert solver._cache is cache
    problem = capture_problem(solver, x0=[2.0])
    assert solver._cache is cache
    np.testing.assert_array_equal(problem.x0, [2.0])
    np.testing.assert_array_equal(problem.x_lower, [1.0])
    np.testing.assert_array_equal(problem.x_upper, [4.0])


def test_simulation_cannot_be_reassigned_even_before_first_solve(solver):
    original = solver.simulation
    other = System(optical_input=Ray(x=Parameter(2).variable(), theta=0, n=1)).build()
    for replacement in (original, other):
        with pytest.raises(AttributeError):
            solver.simulation = replacement
        assert solver.simulation is original


@pytest.mark.filterwarnings("ignore:Output seems independent of input\\.:UserWarning")
@pytest.mark.parametrize("use_jit", [False, True])
def test_repeated_guesses_and_changed_bounds_reuse_core_and_restore_constraint(use_jit):
    parameter = Parameter(0.5).variable(bounds=(0, 3))
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.target(parameter - 2)
    solver.require(Constraint(parameter, upper_bound=1, label="needed after relaxation"))
    restrictive = capture_problem(solver, jit=use_jit, x_upper=[0.75])
    cache, callbacks = solver._cache, solver._cache.callbacks
    assert restrictive.constraints(xp.array([0.5])).shape == (0,)
    relaxed = capture_problem(solver, jit=use_jit, x0=[0.6], x_upper=[3])
    assert solver._cache is cache
    assert cache.callbacks is callbacks
    np.testing.assert_allclose(relaxed.constraints(xp.array([2.])), [2])
    np.testing.assert_allclose(relaxed.jacobian(xp.array([2.])), [[1]])
    np.testing.assert_allclose(relaxed.constraint_hessian(xp.array([2.]), xp.array([5.])), [[0]])
    np.testing.assert_array_equal(relaxed.constraint_upper, [1])
    assert restrictive.constraints(xp.array([2.])).shape == (0,)


@pytest.mark.parametrize("use_jit", [False, True])
def test_vector_duplicates_select_rows_and_expand_hessian_weights(use_jit):
    parameter = Parameter(0.5).variable(bounds=(-3, 3))
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    vector = Literal(np.array([1., 2.])) * parameter ** 2
    solver.target(parameter ** 2)
    solver.require(Constraint(vector, upper_bound=10),
                   Constraint(vector, upper_bound=4),
                   Constraint(parameter, lower_bound=-3))
    problem = capture_problem(solver, jit=use_jit)
    np.testing.assert_allclose(problem.constraints(xp.array([1.5])), [2.25, 4.5])
    np.testing.assert_allclose(problem.jacobian(xp.array([1.5])), [[3], [6]])
    np.testing.assert_allclose(problem.constraint_hessian(xp.array([1.5]), xp.array([3., 5.])), [[26]])
    np.testing.assert_allclose(problem.constraint_upper, [4, 4])


def test_jax_core_does_not_retrace_for_new_guesses_or_bounds(monkeypatch):
    if current_backend() != "jax":
        pytest.skip("JAX executable tracing contract")
    parameter = Parameter(0.5).variable(bounds=(0, 3))
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.target(lambda result: result.final.x ** 2)
    solver.require(Constraint(lambda result: result.final.x ** 2, upper_bound=2))
    traces = []
    original = solver.simulation.run

    def record(theta):
        traces.append(1)
        return original(theta)

    monkeypatch.setattr(solver.simulation, "run", record)
    capture_problem(solver, jit=True)
    core = solver._cache.callbacks

    def evaluate_all(theta):
        for name in ("objective", "gradient", "objective_hessian", "constraints", "jacobian"):
            np.asarray(getattr(core, name)(theta))
        np.asarray(core.constraint_hessian(theta, xp.array([2.])))

    evaluate_all(xp.array([0.5]))
    count = len(traces)
    capture_problem(solver, jit=True, x0=[0.75], x_upper=[2])
    assert solver._cache.callbacks is core
    evaluate_all(xp.array([0.75]))
    assert len(traces) == count


@pytest.mark.parametrize("operation", ["target", "require"])
def test_model_definitions_freeze_after_preparation(operation):
    parameter = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.target(parameter - 1)
    capture_problem(solver)
    entry = parameter if operation == "target" else parameter >= 0
    with pytest.raises(RuntimeError, match="fixed.*new Solver"):
        getattr(solver, operation)(entry)


def test_jit_defaults_to_true_and_changing_it_rebuilds_the_cache(solver):
    capture_problem(solver)
    assert solver._cache_jit is True
    default_cache = solver._cache
    capture_problem(solver, x0=[2.])
    assert solver._cache is default_cache
    capture_problem(solver, jit=False)
    assert solver._cache_jit is False and solver._cache is not default_cache
    eager_cache = solver._cache
    capture_problem(solver, jit=False, x0=[2.])
    assert solver._cache is eager_cache
    capture_problem(solver)
    assert solver._cache_jit is True and solver._cache is not eager_cache


@pytest.mark.parametrize("invalid", [None, 1, object()])
def test_solver_rejects_nonmodel_objects(invalid):
    with pytest.raises(TypeError, match="Simulation or System"):
        Solver(invalid)
