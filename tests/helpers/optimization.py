"""Analytical optimization problems shared by adapter and scaling tests."""

from __future__ import annotations

import numpy as np
from diffractix.solver.problem import Problem


def make_problem(
    *,
    gradient=None,
    jacobian=None,
) -> Problem:
    return Problem(
        x0=np.array([1.0]),
        x_lower=np.array([0.0]),
        x_upper=np.array([2.0]),
        objective=lambda x: x[0] ** 2,
        gradient=gradient or (lambda x: np.array([2.0 * x[0]])),
        objective_hessian=lambda x: np.array([[2.0]]),
        constraints=lambda x: np.array([]),
        jacobian=jacobian or (lambda x: np.empty((0, 1))),
        constraint_hessian=lambda x, multipliers: np.zeros((1, 1)),
        constraint_lower=np.array([]),
        constraint_upper=np.array([]),
    )


def make_constrained_problem() -> Problem:
    return Problem(
        x0=np.array([1.0, 2.0]),
        x_lower=np.array([-5.0, -6.0]),
        x_upper=np.array([5.0, 6.0]),
        objective=lambda x: x[0] ** 2 + 3.0 * x[1] ** 2,
        gradient=lambda x: np.array([2.0 * x[0], 6.0 * x[1]]),
        objective_hessian=lambda x: np.array([[2.0, 0.0], [0.0, 6.0]]),
        constraints=lambda x: np.array(
            [
                x[0] + x[1],
                x[0] - x[1],
                x[0] ** 2,
                x[1] ** 2,
            ]
        ),
        jacobian=lambda x: np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [2.0 * x[0], 0.0],
                [0.0, 2.0 * x[1]],
            ]
        ),
        constraint_hessian=lambda x, multipliers: np.array(
            [
                [2.0 * multipliers[2], 0.0],
                [0.0, 2.0 * multipliers[3]],
            ]
        ),
        constraint_lower=np.array([2.0, 3.0, -np.inf, -4.0]),
        constraint_upper=np.array([5.0, 3.0, 10.0, np.inf]),
    )


def make_scaling_problem() -> Problem:
    return Problem(
        x0=np.array([1.0, 2.0]),
        x_lower=np.array([-3.0, 0.0]),
        x_upper=np.array([5.0, 6.0]),
        objective=lambda x: x[0] ** 2 + x[0] * x[1] + 3.0 * x[1] ** 2,
        gradient=lambda x: np.array([2.0 * x[0] + x[1], x[0] + 6.0 * x[1]]),
        objective_hessian=lambda x: np.array([[2.0, 1.0], [1.0, 6.0]]),
        constraints=lambda x: np.array([x[0] ** 2 + x[0] * x[1], x[1] ** 2 - x[0], 7.0]),
        jacobian=lambda x: np.array([
            [2.0 * x[0] + x[1], x[0]], [-1.0, 2.0 * x[1]], [0.0, 0.0],
        ]),
        constraint_hessian=lambda x, v: np.array([
            [2.0 * v[0], v[0]], [v[0], 2.0 * v[1]],
        ]),
        constraint_lower=np.array([0.0, -np.inf, 7.0]),
        constraint_upper=np.array([32.0, 24.0, 7.0]),
    )


def quadratic_problem() -> Problem:
    return Problem(
        x0=np.array([0.0]),
        x_lower=np.array([-5.0]),
        x_upper=np.array([5.0]),
        objective=lambda x: (x[0] - 2.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0)]),
        objective_hessian=lambda x: np.array([[2.0]]),
        constraints=lambda x: np.array([]),
        jacobian=lambda x: np.empty((0, 1)),
        constraint_hessian=lambda x, multipliers: np.zeros((1, 1)),
        constraint_lower=np.array([]),
        constraint_upper=np.array([]),
    )


def constrained_problem() -> Problem:
    return Problem(
        x0=np.array([1.0, 1.0]),
        x_lower=np.array([0.0, 0.0]),
        x_upper=np.array([4.0, 4.0]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        objective_hessian=lambda x: np.array([[2.0, 0.0], [0.0, 2.0]]),
        constraints=lambda x: np.array([x[0] + x[1]]),
        jacobian=lambda x: np.array([[1.0, 1.0]]),
        constraint_hessian=lambda x, multipliers: np.zeros((2, 2)),
        constraint_lower=np.array([2.0]),
        constraint_upper=np.array([np.inf]),
    )
