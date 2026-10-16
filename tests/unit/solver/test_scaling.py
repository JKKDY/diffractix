"""Variable, constraint, and derivative scaling transformations."""

from __future__ import annotations

import numpy as np
import pytest
from diffractix.solver import OptimizationResult
from diffractix.solver.problem import Problem, Scaling
from tests.helpers.optimization import make_problem, make_scaling_problem


def test_scaling_round_trip_preserves_problem_and_result_values():
    problem = make_problem()
    scaling = problem.scaling
    assert isinstance(scaling, Scaling)
    raw = object()
    result = OptimizationResult(
        x=np.array([1.5]),
        success=True,
        cost=2.25,
        message="done",
        scaling=scaling,
        raw=raw,
    )

    scaled = problem.scale()
    np.testing.assert_array_equal(scaled.x0, [0.0])
    np.testing.assert_array_equal(scaled.x_lower, [-1.0])
    np.testing.assert_array_equal(scaled.x_upper, [1.0])
    assert scaled.objective(np.array([0.5])) == pytest.approx(2.25)

    restored = scaled.unscale()
    for name in ("x0", "x_lower", "x_upper", "constraint_lower", "constraint_upper"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(problem, name))
    x = np.array([1.5])
    for name in ("objective", "gradient", "objective_hessian", "constraints", "jacobian"):
        np.testing.assert_allclose(getattr(restored, name)(x), getattr(problem, name)(x))
    np.testing.assert_allclose(
        restored.constraint_hessian(x, np.array([])),
        problem.constraint_hessian(x, np.array([])),
    )

    scaled_result = scaling.scale(result)
    np.testing.assert_array_equal(scaled_result.x, [0.5])
    restored_result = scaling.unscale(scaled_result)
    np.testing.assert_array_equal(restored_result.x, result.x)
    np.testing.assert_array_equal(result.x, [1.5])
    assert restored_result.raw is raw
    assert restored_result.scaling is scaling
    assert restored_result.cost == result.cost
    assert restored_result.success == result.success
    assert restored_result.message == result.message


def test_variable_constraint_scaling_initialization_and_zero_sensitivity():
    problem = make_scaling_problem()
    scaling = problem.scaling

    np.testing.assert_allclose(scaling.x_center, [1.0, 2.0])
    np.testing.assert_allclose(scaling.x_scale, [4.0, 3.0])
    np.testing.assert_allclose(scaling.constraint_center, [0.0, 0.0, 0.0])
    # Normalized Jacobian rows are [16, 3], [-4, 12], and [0, 0].
    # The constant constraint uses the fallback scale of one.
    np.testing.assert_allclose(scaling.constraint_scale, [16.0, 12.0, 1.0])


def test_variable_constraint_scaling_transforms_bounds_and_values():
    scaled = make_scaling_problem().scale()

    np.testing.assert_allclose(scaled.x0, [0.0, 0.0])
    np.testing.assert_allclose(scaled.x_lower, [-1.0, -2.0 / 3.0])
    np.testing.assert_allclose(scaled.x_upper, [1.0, 4.0 / 3.0])
    np.testing.assert_allclose(scaled.constraint_lower, [0.0, -np.inf, 7.0])
    np.testing.assert_allclose(scaled.constraint_upper, [2.0, 2.0, 7.0])

    # y = [1/2, -1/3] represents physical x = [3, 1].
    y = np.array([0.5, -1.0 / 3.0])
    np.testing.assert_allclose(scaled.objective(y), 15.0)
    np.testing.assert_allclose(scaled.constraints(y), [0.75, -1.0 / 6.0, 7.0])


def test_variable_constraint_scaling_derivatives_and_multipliers():
    scaled = make_scaling_problem().scale()
    y = np.array([0.5, -1.0 / 3.0])

    np.testing.assert_allclose(scaled.gradient(y), [28.0, 27.0])
    np.testing.assert_allclose(scaled.objective_hessian(y), [[32.0, 12.0], [12.0, 54.0]])
    np.testing.assert_allclose(scaled.jacobian(y), [
        [1.75, 9.0 / 16.0], [-1.0 / 3.0, 0.5], [0.0, 0.0],
    ])
    # Numerical multipliers [8, 6, 5] become physical [1/2, 1/2, 5].
    np.testing.assert_allclose(
        scaled.constraint_hessian(y, np.array([8.0, 6.0, 5.0])),
        [[16.0, 6.0], [6.0, 9.0]],
    )


def test_variable_constraint_scaling_problem_round_trip():
    problem = make_scaling_problem()
    restored = problem.scale().unscale()

    for name in ("x0", "x_lower", "x_upper", "constraint_lower", "constraint_upper"):
        np.testing.assert_allclose(getattr(restored, name), getattr(problem, name))
    x = np.array([3.0, 1.0])
    for name in ("objective", "gradient", "objective_hessian", "constraints", "jacobian"):
        np.testing.assert_allclose(getattr(restored, name)(x), getattr(problem, name)(x))
    multipliers = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        restored.constraint_hessian(x, multipliers),
        problem.constraint_hessian(x, multipliers),
    )


def test_variable_constraint_scaling_result_returns_physical_coordinates():
    problem = make_scaling_problem()
    y = np.array([0.5, -1.0 / 3.0])
    raw = {"x": y.copy(), "diagnostic": "backend coordinates"}
    result = OptimizationResult(y.copy(), True, 15.0, "done", raw=raw, scaling=problem.scaling)

    physical = problem.scaling.unscale(result)

    np.testing.assert_allclose(physical.x, [3.0, 1.0])
    np.testing.assert_allclose(result.x, y)
    np.testing.assert_allclose(raw["x"], y)
    assert physical.raw is raw
    assert physical.scaling is problem.scaling
    assert physical.cost == result.cost
    assert physical.success == result.success


def test_scaling_mixed_unbounded_and_zero_width_dimensions_owns_immutable_metadata():
    initial = np.array([0.008, -2.0, 0.0, 3.0])
    problem = Problem(
        x0=initial,
        x_lower=np.array([0.004, -np.inf, -np.inf, 3.0]),
        x_upper=np.array([0.012, np.inf, 1.0, 3.0]),
        objective=lambda x: np.dot(x, x),
        gradient=lambda x: 2 * x,
        objective_hessian=lambda x: 2 * np.eye(4),
        constraints=lambda x: np.array([x[0] + 2 * x[1] - x[2], 7.0]),
        jacobian=lambda x: np.array([[1.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        constraint_hessian=lambda x, v: np.zeros((4, 4)),
        constraint_lower=np.array([-np.inf, 7.0]),
        constraint_upper=np.array([0.0, 7.0]),
    )
    scaling = problem.scaling
    np.testing.assert_allclose(scaling.x_scale, [0.004, 2.0, 1.0, 3.0])
    np.testing.assert_allclose(scaling.constraint_scale, [4.0, 1.0])
    scaled = problem.scale()
    np.testing.assert_allclose(scaled.x_lower, [-1.0, -np.inf, -np.inf, 0.0])
    np.testing.assert_allclose(scaled.x_upper, [1.0, np.inf, 1.0, 0.0])
    np.testing.assert_allclose(scaled.constraints(scaled.x0), [-0.998, 7.0])

    initial[:] = 100.0
    np.testing.assert_allclose(scaling.x_center, [0.008, -2.0, 0.0, 3.0])
    for values in (scaling.x_center, scaling.x_scale, scaling.constraint_center, scaling.constraint_scale):
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0
