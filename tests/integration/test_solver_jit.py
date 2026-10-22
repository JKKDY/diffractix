"""Exercise Solver's JIT opt-in under explicitly selected numerical backends."""
from unittest.mock import patch

import numpy as np
import pytest

from diffractix import GaussianBeam, Parameter, RayBundle, Solver, Space, System, ThinLens
from diffractix.backend import current_backend, hessian, jacobian, jit, xp
from diffractix.solver import Constraint, Objective
from diffractix.solver import solver as solver_module
from tests.helpers.backend import JAX_AVAILABLE, capture_problem, run_python


_CALLBACK_NAMES = (
    "objective", "gradient", "objective_hessian",
    "constraints", "jacobian", "constraint_hessian",
)


@pytest.fixture(params=[
    "autograd",
    pytest.param("jax", marks=pytest.mark.skipif(
        not JAX_AVAILABLE, reason="Optional JAX numerical backend is not installed",
    )),
])
def numerical_backend(request):
    return request.param


def _run_in_backend(check, backend, **kwargs):
    # Selection is process-wide: never reconfigure the parent pytest process.
    if current_backend() == backend:
        check(**kwargs)
        return
    assert run_python(
        f"from tests.integration.test_solver_jit import {check.__name__}\n"
        f"{check.__name__}(**{kwargs!r})\nprint('ok')", backend,
    ) == "ok"


def _hybrid_solver():
    p = Parameter(0.6).variable(bounds=(0.2, 1.5))
    q = Parameter(0.8).variable(bounds=(0.3, 2.0))
    first, second = Space(d=p), Space(d=q)
    source = RayBundle(x=np.array([1.0, 2.0]), theta=np.array([2.0, -1.0]))
    solver = Solver(System(first, second, input=source))
    solver.target(
        Objective(lambda ctx: q**2 + ctx.after(first).x - 2, weight=2.0),
        # Local AST order differs from canonical Simulation order (p, q).
        Objective(q + p - 1, weight=0.5),
    )
    solver.require(
        Constraint(lambda ctx: p * ctx.z_after(second), lower_bound=0.4, upper_bound=4),
        Constraint(lambda ctx: q**2 + ctx.after(first).x, lower_bound=-1, upper_bound=6),
    )
    assert solver.simulation.graph.variables == (p, q)
    return solver


def _check_dispatch():
    solver = _hybrid_solver()
    for kwargs, expected_count in (
        ({}, 0), ({"use_jit": False}, 0), ({"use_jit": True}, 6),
        ({"use_jit": True}, 0), ({"use_jit": False}, 0),
    ):
        wrapped = []

        def record(function):
            result = jit(function)
            wrapped.append((function, result))
            return result

        with patch.object(solver_module, "jit", side_effect=record):
            capture_problem(solver, **kwargs)
        assert len(wrapped) == expected_count
        for name, (original, result) in zip(_CALLBACK_NAMES, wrapped):
            assert getattr(solver._cache.callbacks, name) is original
            assert getattr(solver._cache.jit_callbacks, name) is result
            if current_backend() == "autograd":
                assert result is original
            else:
                # JAX's public lowering API distinguishes real JIT from a no-op.
                assert callable(result.lower)


def test_solver_jit_is_opt_in_and_wraps_all_six_callbacks(numerical_backend):
    _run_in_backend(_check_dispatch, numerical_backend)


def _check_cache_reuse():
    solver = _hybrid_solver()
    p, _ = solver.simulation.graph.variables
    solver.require(p >= 0.5)
    trace_counts = []

    def record(function):
        index = len(trace_counts)
        trace_counts.append(0)

        def counted(*args):
            trace_counts[index] += 1
            return function(*args)

        return jit(counted)

    with patch.object(solver_module, "jit", side_effect=record), patch.object(
        solver, "_build_cache", wraps=solver._build_cache,
    ) as build_cache:
        first = capture_problem(solver, use_jit=True)
        cache = solver._cache
        eager = cache.callbacks
        compiled = cache.jit_callbacks
        original_functions = tuple(getattr(eager, name) for name in _CALLBACK_NAMES)
        jit_functions = tuple(getattr(compiled, name) for name in _CALLBACK_NAMES)

        def warm_core(theta):
            # Warm all six full callbacks, not the per-solve row selectors.
            for name, function in zip(_CALLBACK_NAMES, jit_functions):
                extra = (xp.ones(cache.constraint_size),) if name == "constraint_hessian" else ()
                np.asarray(function(xp.array(theta), *extra))

        warm_core([0.6, 0.8])
        initial_traces = trace_counts.copy()
        if current_backend() == "jax":
            assert initial_traces == [1] * 6
        assert first.n_constraints == 4

        for use_jit, lower, point, expected_rows in (
            (True, [0.7, 0.4], [0.9, 0.5], 3),
            (True, [0.2, 0.3], [0.6, 1.0], 4),
            (False, [0.7, 0.4], [0.9, 0.5], 3),
        ):
            problem = capture_problem(
                solver, use_jit=use_jit, x0=point, x_lower=lower, x_upper=[1.5, 1.6],
            )
            assert problem.n_constraints == expected_rows
            assert solver._cache is cache
            assert cache.callbacks is eager
            assert cache.jit_callbacks is compiled
            for name, original, wrapped in zip(_CALLBACK_NAMES, original_functions, jit_functions):
                assert getattr(cache.callbacks, name) is original
                assert getattr(cache.jit_callbacks, name) is wrapped
            warm_core(point)
            if current_backend() == "jax":
                assert trace_counts == initial_traces, "The complete numerical core retraced"

        assert build_cache.call_count == 1
        assert len(trace_counts) == 6


def test_solver_jit_reuses_traces_for_new_same_shaped_values(numerical_backend):
    _run_in_backend(_check_cache_reuse, numerical_backend)


def _weighted_hessian(multipliers):
    a, b, c = multipliers
    return np.array([[2*a, a], [a, 2*(b+c)]])


def _check_callbacks():
    solver = _hybrid_solver()
    for use_jit in (False, True):
        physical = capture_problem(solver, use_jit=use_jit)
        scaled = physical.scale()
        scaling = physical.scaling
        sx, sc = scaling.x_scale, scaling.constraint_scale
        assert physical.n_constraints == 3  # One scalar and one two-ray vector.
        np.testing.assert_allclose(physical.constraint_lower, [0.4, -1, -1])
        np.testing.assert_allclose(physical.constraint_upper, [4, 6, 6])

        for point in ([0.6, 0.8], [0.9, 0.5]):
            theta = xp.array(point)
            p, q = point
            # Independent free-space ray formula: x_out = [1+2p, 2-p].
            a, b, s = 2*p + q*q - 1, q*q - p, p + q - 1
            da, db = np.array([2, 2*q]), np.array([-1, 2*q])
            cost = 2*(a*a + b*b) + 0.5*s*s
            gradient = 4*(a*da + b*db) + s
            objective_hessian = 4*(np.outer(da, da) + np.outer(db, db)
                                      + np.diag([0, 2*(a+b)])) + np.ones((2, 2))
            values = np.array([p*(p+q), 1+2*p+q*q, 2-p+q*q])
            constraint_jacobian = np.array([[2*p+q, p], [2, 2*q], [-1, 2*q]])
            y = xp.array((np.array(point) - scaling.x_center) / sx)

            for problem, argument, expected in (
                (physical, theta, (cost, gradient, objective_hessian, values, constraint_jacobian)),
                (scaled, y, (cost, sx*gradient, sx[:, None]*objective_hessian*sx[None, :],
                             values/sc, constraint_jacobian*sx[None, :]/sc[:, None])),
            ):
                callbacks = (problem.objective, problem.gradient, problem.objective_hessian,
                             problem.constraints, problem.jacobian)
                for callback, reference in zip(callbacks, expected):
                    actual = callback(argument)
                    assert np.shape(actual) == np.shape(reference)
                    np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-11)

            # Changing multipliers must change the compiled weighted Hessian.
            for weights in ([0.7, -1.2, 0.3], [-0.2, 0.4, 1.5]):
                multipliers = xp.array(weights)
                np.testing.assert_allclose(
                    physical.constraint_hessian(theta, multipliers),
                    _weighted_hessian(weights), rtol=2e-12, atol=1e-12,
                )
                np.testing.assert_allclose(
                    scaled.constraint_hessian(y, multipliers),
                    sx[:, None]*_weighted_hessian(np.array(weights)/sc)*sx[None, :],
                    rtol=2e-12, atol=1e-12,
                )

            np.testing.assert_allclose(jacobian(physical.gradient)(theta), objective_hessian,
                                       rtol=2e-12, atol=1e-11)
            # Differentiate through an already compiled second derivative.
            fourth = hessian(lambda t: physical.objective_hessian(t)[1, 1])(theta)
            np.testing.assert_allclose(fourth, [[0, 0], [0, 96]], rtol=2e-12, atol=1e-11)


def test_solver_jit_values_derivatives_scaling_and_higher_order_ad(numerical_backend):
    _run_in_backend(_check_callbacks, numerical_backend)


def _check_reduced_callbacks():
    solver = _hybrid_solver()
    p, _ = solver.simulation.graph.variables
    shared = p**2
    solver.require(
        Constraint(p, lower_bound=0.5, label="minimum p"),
        Constraint(shared, lower_bound=0.1, label="squared minimum"),
        Constraint(shared, upper_bound=2.2, label="squared maximum"),
        Constraint(shared, upper_bound=2.2, label="duplicate squared maximum"),
    )

    # Restrict, relax, then restrict again: retained rows must not be cached by bounds.
    for lower, has_affine_row in ((0.55, False), (0.2, True), (0.55, False)):
        for use_jit in (False, True):
            problem = capture_problem(solver, use_jit=use_jit, x_lower=[lower, 0.3])
            expected_lower = [0.4, -1, -1] + ([0.5] if has_affine_row else []) + [0.1, -np.inf]
            expected_upper = [4, 6, 6] + ([np.inf] if has_affine_row else []) + [np.inf, 2.2]
            # capture_problem round-trips through physical/numerical scaling.
            np.testing.assert_allclose(problem.constraint_lower, expected_lower, rtol=1e-14, atol=0)
            np.testing.assert_allclose(problem.constraint_upper, expected_upper, rtol=1e-14, atol=0)
            assert problem.n_constraints == len(expected_lower)

            for point in ([0.6, 0.8], [0.9, 0.5]):
                a, b = point
                theta = xp.array(point)
                expected_values = [a*(a+b), 1+2*a+b*b, 2-a+b*b]
                expected_jacobian = [[2*a+b, a], [2, 2*b], [-1, 2*b]]
                if has_affine_row:
                    expected_values.append(a)
                    expected_jacobian.append([1, 0])
                expected_values.extend([a*a, a*a])
                expected_jacobian.extend([[2*a, 0], [2*a, 0]])
                np.testing.assert_allclose(problem.constraints(theta), expected_values, atol=1e-12)
                np.testing.assert_allclose(problem.jacobian(theta), expected_jacobian, atol=1e-12)

                # The same full-vector entry supplies both retained squared intervals.
                weights = xp.array([0.7, -1.2, 0.3] + ([0.4] if has_affine_row else []) + [0.5, -0.8])
                np.testing.assert_allclose(
                    problem.constraint_hessian(theta, weights),
                    [[0.8, 0.7], [0.7, -1.8]], atol=1e-12,
                )
                pieces = [np.array([[2, 1], [1, 0]]), np.diag([0, 2]), np.diag([0, 2])]
                if has_affine_row:
                    pieces.append(np.zeros((2, 2)))
                pieces.extend([np.diag([2, 0]), np.diag([2, 0])])
                np.testing.assert_allclose(
                    jacobian(lambda v: problem.constraint_hessian(theta, v))(weights),
                    np.stack(pieces, axis=-1), atol=1e-12,
                )


def test_reduced_vector_constraints_and_duplicate_hessian_weights(numerical_backend):
    _run_in_backend(_check_reduced_callbacks, numerical_backend)


def _check_repeated_solves():
    distance = Parameter(1.5).variable(bounds=(1.0, 4.0))
    space = Space(d=distance)
    simulation = System(space, input=GaussianBeam.from_waist(0.001, 1e-6)).build()
    solver = Solver(simulation)
    solver.target(lambda ctx: ctx.z_after(space) - 0.2)
    solver.require(Constraint(distance, lower_bound=0.5, label="minimum length"))
    requirements = simulation.requirements
    metadata = simulation.parameter_info
    cases = (
        ({}, [1.5], [1.0], [4.0], 1.0, 0),
        ({"x0": [2.5]}, [2.5], [1.0], [4.0], 1.0, 0),
        ({"x0": [1.2], "x_lower": [0.0], "x_upper": [3.0]}, [1.2], [0.0], [3.0], 0.5, 1),
        ({"x0": [1.6], "x_lower": [1.2], "x_upper": [2.0]}, [1.6], [1.2], [2.0], 1.2, 0),
        ({}, [1.5], [1.0], [4.0], 1.0, 0),
    )
    with patch.object(solver, "_build_cache", wraps=solver._build_cache) as build_cache:
        eager_solutions = []
        for use_jit in (False, True):
            for index, (overrides, start, lower, upper, optimum, rows) in enumerate(cases):
                overrides = {name: np.array(value) for name, value in overrides.items()}
                solution = solver.solve(method="SLSQP", use_jit=use_jit, **overrides)
                assert solution.success, solution.message
                assert solution.feasible
                np.testing.assert_allclose(solution.x, [optimum], atol=1e-8)
                np.testing.assert_allclose(solution.cost, (optimum - 0.2)**2, atol=1e-10)
                np.testing.assert_array_equal(solution._problem.x0, start)
                np.testing.assert_array_equal(solution._problem.x_lower, lower)
                np.testing.assert_array_equal(solution._problem.x_upper, upper)
                assert solution._problem.n_constraints == rows
                for name, supplied in overrides.items():
                    expected = {"x0": start, "x_lower": lower, "x_upper": upper}[name]
                    np.testing.assert_array_equal(supplied, expected)
                if use_jit:
                    np.testing.assert_allclose(solution.x, eager_solutions[index].x, atol=1e-10)
                    np.testing.assert_allclose(solution.cost, eager_solutions[index].cost, atol=1e-10)
                else:
                    eager_solutions.append(solution)
        assert build_cache.call_count == 1

    # Results and declarative defaults must survive all later solve overrides.
    np.testing.assert_array_equal(eager_solutions[0].x, [1.0])
    np.testing.assert_array_equal(simulation.initial_values, [1.5])
    assert simulation.requirements is requirements
    assert simulation.parameter_info is metadata
    assert distance.value == 1.5
    assert (metadata[id(distance)].lower_bound, metadata[id(distance)].upper_bound) == (1.0, 4.0)


@pytest.mark.scipy
def test_repeated_solves_respect_overrides_without_changing_defaults(numerical_backend):
    _run_in_backend(_check_repeated_solves, numerical_backend)


def _check_empty_constraints(use_jit):
    import warnings

    focal = Parameter(1.5).variable(bounds=(0.2, 3.0))
    solver = Solver(System(ThinLens(f=focal), input=GaussianBeam.from_waist(0.001, 1e-6)))
    solver.target(focal - 1)
    problem = capture_problem(solver, use_jit=use_jit)
    theta = xp.array([1.2])
    assert problem.n_constraints == 0
    np.testing.assert_allclose(problem.objective(theta), 0.04)
    np.testing.assert_allclose(problem.gradient(theta), [0.4])
    np.testing.assert_allclose(problem.objective_hessian(theta), [[2]])
    assert problem.constraints(theta).shape == (0,)
    # Empty constraint derivatives are intentionally independent of theta.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output seems independent of input\\.")
        assert problem.jacobian(theta).shape == (0, 1)
        np.testing.assert_array_equal(problem.constraint_hessian(theta, xp.array([])), [[0]])


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
def test_solver_jit_handles_no_constraints(numerical_backend, use_jit):
    _run_in_backend(_check_empty_constraints, numerical_backend, use_jit=use_jit)


def _check_real_solve():
    wavelength, waist = 1e-6, 1e-4
    distance = Parameter(0.15).variable(bounds=(0.05, 0.4))
    space = Space(d=distance)
    simulation = System(space, input=GaussianBeam.from_waist(waist, wavelength)).build()
    target_w = np.sqrt(waist**2 + (wavelength*0.3/(np.pi*waist))**2)
    solver = Solver(simulation)
    solver.target(value=lambda ctx: ctx.after(space).w, target=target_w,
                  scale=target_w, label="radius")
    solver.require(Constraint(lambda ctx: ctx.z_after(space)**2, upper_bound=0.04,
                              label="maximum squared length"))
    initial = np.array(simulation.initial_values, copy=True)
    requirements = simulation.requirements
    options = {"ftol": 1e-12, "maxiter": 100}
    solutions = []
    # The unconstrained radius target lies at 0.3; the active constraint forces 0.2.
    expected_w = np.sqrt(waist**2 + (wavelength*0.2/(np.pi*waist))**2)
    for use_jit in (False, True):
        solution = solver.solve(method="SLSQP", options=options, use_jit=use_jit)
        assert solution.success, solution.message
        assert solution.feasible
        np.testing.assert_allclose(solution.x, [0.2], atol=1e-7, rtol=0)
        np.testing.assert_allclose(solution.cost, (expected_w/target_w - 1)**2, atol=1e-8)
        np.testing.assert_allclose(solution.run().final.w, expected_w, atol=1e-10)
        assert solution.targets[0].objective.label == "radius"
        assert solution.constraints[-1].constraint.label == "maximum squared length"
        np.testing.assert_allclose(solution.constraints[-1].value, 0.04, atol=1e-8)
        solutions.append(solution)
    np.testing.assert_allclose(solutions[0].x, solutions[1].x, atol=1e-8)
    np.testing.assert_array_equal(simulation.initial_values, initial)
    assert simulation.requirements is requirements
    assert distance.value == 0.15
    assert options == {"ftol": 1e-12, "maxiter": 100}


@pytest.mark.scipy
def test_solver_jit_real_constrained_gaussian_optimization(numerical_backend):
    _run_in_backend(_check_real_solve, numerical_backend)


def _check_dynamic_control_flow():
    distance = Parameter(0.6).variable(bounds=(0.2, 1.5))
    space = Space(d=distance)
    solver = Solver(System(space, input=GaussianBeam.from_waist(0.001, 1e-6)))

    def objective(ctx):
        if ctx.z_after(space) > 0.5:
            return distance**2
        return distance + 1

    solver.target(objective)
    for use_jit in (False, True):
        problem = capture_problem(solver, use_jit=use_jit)
        if use_jit and current_backend() == "jax":
            from jax.errors import TracerBoolConversionError

            # JIT is opt-in, not an automatic rewrite of runtime Python branches.
            for callback in (problem.objective, problem.gradient):
                with pytest.raises(TracerBoolConversionError):
                    callback(xp.array([0.6]))
        else:
            for p, cost, derivative in ((0.6, 0.6**4, 4*0.6**3), (0.3, 1.3**2, 2*1.3)):
                np.testing.assert_allclose(problem.objective(xp.array([p])), cost)
                np.testing.assert_allclose(problem.gradient(xp.array([p])), [derivative])


def test_solver_jit_does_not_silently_change_dynamic_python_branches(numerical_backend):
    _run_in_backend(_check_dynamic_control_flow, numerical_backend)
