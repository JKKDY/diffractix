"""Exercise Solver's JIT opt-in under explicitly selected numerical backends."""
from unittest.mock import patch

import numpy as np
import pytest

from diffractix import GaussianBeam, Parameter, RayBundle, Solver, Space, System, ThinLens
from diffractix.backend import current_backend, hessian, jacobian, jit, xp
from diffractix.solver import Constraint, Objective
from diffractix.solver import solver as solver_module
from tests.helpers.backend import JAX_AVAILABLE, capture_problem, run_python


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
    for kwargs, expected_count in (({}, 0), ({"use_jit": False}, 0), ({"use_jit": True}, 6)):
        wrapped = []

        def record(function):
            result = jit(function)
            wrapped.append((function, result))
            return result

        with patch.object(solver_module, "jit", side_effect=record):
            capture_problem(solver, **kwargs)
        assert len(wrapped) == expected_count
        for original, result in wrapped:
            if current_backend() == "autograd":
                assert result is original
            else:
                # JAX's public lowering API distinguishes real JIT from a no-op.
                assert callable(result.lower)


def test_solver_jit_is_opt_in_and_wraps_all_six_callbacks(numerical_backend):
    _run_in_backend(_check_dispatch, numerical_backend)


def _check_cache_reuse():
    solver = _hybrid_solver()
    run = solver.simulation.run
    calls = []

    def record_run(theta):
        calls.append(None)
        return run(theta)

    solver.simulation.run = record_run
    for use_jit in (False, True):
        problem = capture_problem(solver, use_jit=use_jit)
        for name in ("objective", "gradient", "objective_hessian", "constraints",
                     "jacobian", "constraint_hessian"):
            callback = getattr(problem, name)
            extra = (xp.array([0.7, -1.2, 0.3]),) if name == "constraint_hessian" else ()
            callback(xp.array([0.6, 0.8]), *extra)
            first_calls = len(calls)
            callback(xp.array([0.9, 0.5]), *extra)
            if use_jit and current_backend() == "jax":
                assert len(calls) == first_calls, f"{name} retraced for same-shaped input"
            else:
                assert len(calls) > first_calls, f"{name} unexpectedly bypassed Python"


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
