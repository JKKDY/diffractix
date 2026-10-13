"""Solver orchestration with real SciPy optimization."""

from __future__ import annotations

import numpy as numpy
import pytest
from diffractix.graph import Parameter
from diffractix.solver import Backend, Objective, Solution, Solver
from tests.helpers.simulation import make_simulation


@pytest.mark.scipy
def test_solver_solve_returns_real_solution_with_scipy():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter - 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert isinstance(solution, Solution)
    assert solution.success
    assert solution.cost == pytest.approx(0.0, abs=1e-10)
    numpy.testing.assert_allclose(solution.x, [2.0], atol=1e-6)


@pytest.mark.scipy
def test_solver_solve_respects_parameter_bounds():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=3.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter - 10.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    numpy.testing.assert_allclose(solution.x, [3.0], atol=1e-6)


@pytest.mark.scipy
def test_solver_solve_enforces_hard_constraint():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter)
    solver.require(parameter >= 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    assert solution.violations == ()
    numpy.testing.assert_allclose(solution.x, [2.0], atol=1e-6)
    assert solution.constraints[0].satisfied


@pytest.mark.scipy
def test_solver_solve_mixed_objectives_constraint_and_bounds():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(
        Objective(parameter - 1.0),
        Objective(parameter - 3.0, weight=3.0),
    )
    solver.require(parameter >= 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    numpy.testing.assert_allclose(solution.x, [2.5], atol=1e-5)
    assert solution.cost == pytest.approx(sum(target.cost for target in solution.targets))
