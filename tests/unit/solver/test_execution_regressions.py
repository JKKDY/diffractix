"""Exercise cache construction and callbacks through the rewritten Solver."""

import numpy as np
import pytest

from diffractix import Constraint, Objective, Parameter, Ray, Solver, System
from diffractix.backend import grad, hessian, jacobian, xp
from diffractix.graph import Literal, Symbol
from tests.helpers.backend import capture_problem


def make_solver(parameter):
    return Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))


@pytest.mark.parametrize("use_jit", [False, True])
def test_runtime_symbols_and_cache_metadata_work_through_normal_preparation(use_jit):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    solver.target(Objective(lambda result: result.final.x - 1, label="target"))
    problem = capture_problem(solver, use_jit=use_jit, x_lower=[-3], x_upper=[3])
    cache = solver._cache
    assert cache.objective_descriptors[0].label == "target"
    assert cache.objective_program.runtime_symbols
    assert cache.callable_objectives == ()
    assert problem.objective(xp.array([2.])) == pytest.approx(1)
    np.testing.assert_allclose(problem.gradient(xp.array([2.])), [2])
    capture_problem(solver, use_jit=use_jit, x0=[0.75], x_lower=[-2], x_upper=[2])
    assert solver._cache is cache


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.filterwarnings("ignore:Output seems independent of input\\.:UserWarning")
def test_constraint_only_problem_has_a_differentiable_floating_objective(use_jit):
    parameter = Parameter(1.5).variable()
    solver = make_solver(parameter)
    solver.require(Constraint(lambda result: result.final.x, lower_bound=1))
    problem = capture_problem(solver, use_jit=use_jit)
    assert problem.objective(xp.array([2.])) == 0
    assert np.asarray(problem.objective(xp.array([2.]))).dtype.kind == "f"
    np.testing.assert_allclose(problem.gradient(xp.array([2.])), [0])
    np.testing.assert_allclose(problem.objective_hessian(xp.array([2.])), [[0]])
    np.testing.assert_allclose(problem.constraints(xp.array([2.])), [2])
    np.testing.assert_allclose(problem.jacobian(xp.array([2.])), [[1]])


def test_dynamic_numeric_branches_work_with_eager_execution():
    parameter = Parameter(0.5).variable()
    system = System(optical_input=Ray(x=parameter, theta=0, n=1))
    solver = Solver(system)

    def residual(result):
        if result.final.x > 0:
            return 2 * result.final.x ** 2
        return 5.

    def constraint(result):
        if result.final.x > 0:
            return result.final.x ** 2
        return -result.final.x

    solver.target(Objective(residual, weight=3))
    solver.require(Constraint(constraint, upper_bound=10))
    problem = capture_problem(solver, use_jit=False)
    parameter.value = 100
    positive, negative = xp.array([2.]), xp.array([-2.])
    assert problem.objective(positive) == pytest.approx(192)
    np.testing.assert_allclose(problem.gradient(positive), [384])
    np.testing.assert_allclose(problem.objective_hessian(positive), [[576]])
    np.testing.assert_allclose(problem.constraints(positive), [4])
    np.testing.assert_allclose(problem.jacobian(positive), [[4]])
    np.testing.assert_allclose(problem.constraint_hessian(positive, xp.array([3.])), [[6]])
    assert problem.objective(negative) == pytest.approx(75)
    np.testing.assert_allclose(problem.gradient(negative), [0])
    np.testing.assert_allclose(problem.constraints(negative), [2])
    np.testing.assert_allclose(problem.jacobian(negative), [[-1]])


def test_dynamic_numeric_arrays_are_flattened():
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)

    def constraint(result):
        if result.final.x > 0:
            return result.final.x - np.array([1., 2.])
        return np.array([3., 4.])

    solver.require(Constraint(constraint, upper_bound=10))
    problem = capture_problem(solver, use_jit=False)
    np.testing.assert_allclose(problem.constraints(xp.array([2.])), [1, 0])
    np.testing.assert_allclose(problem.jacobian(xp.array([2.])), [[1], [1]])
    np.testing.assert_allclose(problem.constraints(xp.array([-2.])), [3, 4])


@pytest.mark.parametrize("kind", ["objective", "constraint"])
@pytest.mark.parametrize("returned", ["parameter", "expression", "literal", "symbol", "external"])
def test_deferred_callables_reject_symbolic_nodes(kind, returned):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    node = {
        "parameter": parameter, "expression": parameter ** 2,
        "literal": Literal(1), "symbol": Symbol("value"), "external": Parameter(1).variable(),
    }[returned]

    def dynamic(result):
        if result.final.x > 0:
            return node
        return 0.

    if kind == "objective":
        solver.target(dynamic)
    else:
        solver.require(Constraint(dynamic))
    with pytest.raises(TypeError, match="numerical scalars or arrays.*symbolic Node"):
        problem = capture_problem(solver, use_jit=False)
        problem.objective(xp.array([0.5]))


@pytest.mark.parametrize("dynamic_kind", ["objective", "constraint"])
def test_requesting_jit_wraps_every_callback_even_with_deferred_callables(dynamic_kind, monkeypatch):
    import diffractix.solver.solver as solver_module

    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)

    def dynamic(result):
        return result.final.x ** 2 if result.final.x > 0 else 5.

    if dynamic_kind == "objective":
        solver.target(dynamic)
        solver.require(Constraint(lambda result: result.final.x, upper_bound=10))
    else:
        solver.target(lambda result: result.final.x - 1)
        solver.require(Constraint(dynamic, upper_bound=10))
    wrapped = []

    def record_jit(function):
        wrapped.append(function)
        return function

    monkeypatch.setattr(solver_module, "jit", record_jit)
    problem = capture_problem(solver, use_jit=True)
    assert len(wrapped) == 6
    theta = xp.array([2.])
    if dynamic_kind == "objective":
        assert problem.objective(theta) == pytest.approx(16)
        np.testing.assert_allclose(problem.gradient(theta), [32])
        np.testing.assert_allclose(problem.constraints(theta), [2])
        np.testing.assert_allclose(problem.jacobian(theta), [[1]])
    else:
        assert problem.objective(theta) == pytest.approx(1)
        np.testing.assert_allclose(problem.gradient(theta), [2])
        np.testing.assert_allclose(problem.constraints(theta), [4])
        np.testing.assert_allclose(problem.jacobian(theta), [[4]])


@pytest.mark.parametrize("kind", ["objective", "constraint"])
def test_deferred_nodes_are_rejected_when_a_later_evaluation_selects_the_branch(kind):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)

    def dynamic(result):
        return result.final.x if result.final.x > 0 else parameter

    if kind == "objective":
        solver.target(dynamic)
    else:
        solver.require(Constraint(dynamic))
    problem = capture_problem(solver, use_jit=False)
    callback = problem.objective if kind == "objective" else problem.constraints
    with pytest.raises(TypeError, match="symbolic Node"):
        callback(xp.array([-0.5]))


def test_traceable_callables_still_compile_nodes_and_runtime_quantities():
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    solver.target(lambda result: parameter ** 2 + result.final.x)
    solver.require(Constraint(lambda result: 2 * parameter + result.final.x, upper_bound=10))
    problem = capture_problem(solver, use_jit=True)
    assert solver._cache.callable_objectives == solver._cache.callable_constraints == ()
    parameter.value = 100
    assert problem.objective(xp.array([2.])) == pytest.approx(36)
    np.testing.assert_allclose(problem.gradient(xp.array([2.])), [60])
    np.testing.assert_allclose(problem.constraints(xp.array([2.])), [6])
    np.testing.assert_allclose(problem.jacobian(xp.array([2.])), [[3]])


@pytest.mark.parametrize("kind", ["objective", "constraint"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("vector", [False, True])
def test_nonfinite_deferred_outputs_are_rejected_with_the_entry_label(kind, value, vector):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)

    def dynamic(result):
        if result.final.x > 0:
            return xp.array([1., value]) if vector else value
        return xp.zeros(2) if vector else 0.

    if kind == "objective":
        solver.target(Objective(dynamic, label="invalid residual"))
    else:
        solver.require(Constraint(dynamic, label="invalid residual"))
    with pytest.raises(ValueError, match="invalid residual.*non-finite"):
        cache = solver._build_cache(False, solver.simulation.initial_values.values)
        cache.callbacks.objective(xp.array([0.5]))


@pytest.mark.parametrize("kind", ["objective", "constraint"])
@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize("numerator", [0., 1., -1.])
def test_nonfinite_runtime_outputs_are_rejected_even_with_jit(kind, use_jit, numerator):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    function = lambda result: numerator / result.final.x
    if kind == "objective":
        solver.target(function)
    else:
        solver.require(Constraint(function))
    cache = solver._build_cache(use_jit, solver.simulation.initial_values.values)
    callback = cache.callbacks.objective if kind == "objective" else cache.callbacks.constraints
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="[Oo]bjective.*non-finite|[Cc]onstraint.*non-finite"):
            callback(xp.array([0.]))


@pytest.mark.parametrize("use_jit", [False, True])
def test_overflow_in_objective_cost_is_rejected(use_jit):
    parameter = Parameter(1).variable()
    solver = make_solver(parameter)
    solver.target(lambda result: result.final.x)
    cache = solver._build_cache(use_jit, solver.simulation.initial_values.values)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="Objective cost.*non-finite"):
            cache.callbacks.objective(xp.array([1e200]))


@pytest.mark.parametrize("kind", ["objective", "constraint"])
def test_jitted_vector_outputs_are_checked_componentwise(kind):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    function = lambda result: Literal(np.array([1., 0.])) / result.final.x
    if kind == "objective":
        solver.target(function)
    else:
        solver.require(Constraint(function))
    cache = solver._build_cache(True, solver.simulation.initial_values.values)
    callback = cache.callbacks.objective if kind == "objective" else cache.callbacks.constraints
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="non-finite"):
            callback(xp.array([0.]))


@pytest.mark.parametrize("initial_shape,later_shape", [((2,), (1, 2)), ((2,), (3,)), ((), (1,)), ((0,), (1,))])
def test_constraint_output_shapes_remain_fixed(initial_shape, later_shape):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)

    def dynamic(result):
        shape = initial_shape if result.final.x > 0 else later_shape
        return xp.full(shape, result.final.x)

    solver.require(Constraint(dynamic, label="changing output"))
    cache = solver._build_cache(False, solver.simulation.initial_values.values)
    with pytest.raises(ValueError, match="changing output.*shape"):
        cache.callbacks.constraints(xp.array([-0.5]))


@pytest.mark.parametrize("use_jit", [False, True])
def test_validation_wrappers_preserve_differentiation(use_jit):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    solver.target(lambda result: result.final.x ** 2)
    solver.require(Constraint(lambda result: result.final.x ** 2))
    cache = solver._build_cache(use_jit, solver.simulation.initial_values.values)
    theta = xp.array([2.])
    np.testing.assert_allclose(grad(cache.callbacks.objective)(theta), [32])
    np.testing.assert_allclose(hessian(cache.callbacks.objective)(theta), [[48]])
    np.testing.assert_allclose(jacobian(cache.callbacks.constraints)(theta), [[4]])


@pytest.mark.parametrize("use_jit", [False, True])
def test_parameter_only_callbacks_do_not_run_simulation(use_jit, monkeypatch):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    solver.target(parameter ** 2)
    solver.require(Constraint(parameter ** 2, upper_bound=10))

    def unexpected_run(*args):
        raise AssertionError("Parameter-only expressions must not propagate")

    monkeypatch.setattr(solver.simulation, "run", unexpected_run)
    cache = solver._build_cache(use_jit, solver.simulation.initial_values.values)
    theta = xp.array([2.])
    assert cache.callbacks.objective(theta) == pytest.approx(16)
    np.testing.assert_allclose(cache.callbacks.gradient(theta), [32])
    np.testing.assert_allclose(cache.callbacks.constraints(theta), [4])
    np.testing.assert_allclose(cache.callbacks.jacobian(theta), [[4]])


@pytest.mark.parametrize("kind", ["objective", "constraint"])
def test_deferred_callables_share_one_result_view(kind):
    parameter = Parameter(0.5).variable()
    solver = make_solver(parameter)
    views = []

    def residual(view):
        if view.final.x > 0:
            views.append(view)
            return view.final.x ** 2
        return 0.

    if kind == "objective":
        solver.target(residual, residual)
    else:
        solver.require(Constraint(residual), Constraint(residual))
    cache = solver._build_cache(False, solver.simulation.initial_values.values)
    views.clear()
    callback = cache.callbacks.objective if kind == "objective" else cache.callbacks.constraints
    value = callback(xp.array([2.]))
    assert len(views) == 2
    assert views[0] is views[1]
    np.testing.assert_allclose(value, 32 if kind == "objective" else [4, 4])
