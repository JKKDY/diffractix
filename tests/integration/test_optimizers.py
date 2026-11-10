"""Real optimization backends; optional packages skip per test."""

from __future__ import annotations

import importlib
from dataclasses import replace
import numpy as np
import pytest
from tests.helpers.optimization import constrained_problem, quadratic_problem


@pytest.mark.scipy
def test_variable_constraint_scaling_scipy_constrained_quadratic():
    from diffractix.solver.optimizers.scipy import solve_scipy

    # Project the unconstrained optimum [2, 1] onto x[0] + x[1] >= 4.
    problem = replace(constrained_problem(), constraint_lower=np.array([4.0]))
    result = solve_scipy(problem.scale(), method="SLSQP", options={"ftol": 1e-12})
    physical = problem.scaling.unscale(result)

    assert physical.success, physical.message
    np.testing.assert_allclose(physical.x, [2.5, 1.5], atol=1e-7)
    np.testing.assert_allclose(physical.cost, 0.5, atol=1e-10)
    values = problem.constraints(physical.x)
    assert np.all(values >= problem.constraint_lower - 1e-8)
    assert np.all(values <= problem.constraint_upper + 1e-8)


@pytest.mark.scipy
def test_real_scipy_backend_smoke():
    pytest.importorskip("scipy")
    from diffractix.solver.optimizers.scipy import solve_scipy

    result = solve_scipy(
        quadratic_problem(),
        method="trust-constr",
        options={"gtol": 1e-10},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [2.0], atol=1e-5)
    assert result.cost == pytest.approx(0.0, abs=1e-10)


@pytest.mark.nlopt
def test_real_nlopt_backend_smoke():
    pytest.importorskip("nlopt")
    backend = importlib.reload(
        importlib.import_module("diffractix.solver.optimizers.nlopt")
    )

    problem = replace(constrained_problem(), constraint_lower=np.array([4.0]))
    result = backend.solve_nlopt(
        problem,
        method="LD_SLSQP",
        options={"xtol_rel": 1e-9},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [2.5, 1.5], atol=1e-5)
    assert result.cost == pytest.approx(0.5, abs=1e-5)
    values = problem.constraints(result.x)
    assert np.all(values >= problem.constraint_lower - 1e-7)
    assert np.all(values <= problem.constraint_upper + 1e-7)


@pytest.mark.ipopt
def test_real_ipopt_backend_smoke():
    pytest.importorskip("cyipopt")
    backend = importlib.reload(
        importlib.import_module("diffractix.solver.optimizers.ipopt")
    )

    problem = replace(constrained_problem(), constraint_lower=np.array([4.0]))
    result = backend.solve_ipopt(
        problem,
        options={"print_level": 0},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [2.5, 1.5], atol=1e-5)
    assert result.cost == pytest.approx(0.5, abs=1e-5)
    values = problem.constraints(result.x)
    assert np.all(values >= problem.constraint_lower - 1e-7)
    assert np.all(values <= problem.constraint_upper + 1e-7)



@pytest.mark.scipy
@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
def test_native_scipy_solves_scaled_mixed_equality_and_inequality_constraints(method):
    from diffractix.solver.problem import Problem
    from diffractix.solver.optimizers.scipy import solve_scipy

    # Projection onto sum >= 4 and difference == 1 is uniquely (2.5, 1.5).
    problem = Problem(
        x0=np.array([2., 1.]), x_lower=np.array([0., 0.]), x_upper=np.array([5., 5.]),
        objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
        gradient=lambda x: 2*(x-np.array([2., 1.])),
        objective_hessian=lambda x: 2*np.eye(2),
        constraints=lambda x: np.array([x[0]+x[1], x[0]-x[1]]),
        jacobian=lambda x: np.array([[1., 1.], [1., -1.]]),
        constraint_hessian=lambda x, multipliers: np.zeros((2, 2)),
        constraint_lower=np.array([4., 1.]), constraint_upper=np.array([np.inf, 1.]),
    )
    options = {"ftol": 1e-12} if method == "SLSQP" else {"gtol": 1e-10}
    result = solve_scipy(problem.scale(), method=method, options=options)
    physical = problem.scaling.unscale(result)
    assert physical.success, physical.message
    np.testing.assert_allclose(physical.x, [2.5, 1.5], atol=2e-4)
    assert abs(physical.x[0]-physical.x[1]-1) < 1e-8
    assert sum(physical.x) >= 4-1e-8
    assert physical.cost == pytest.approx(0.5, abs=3e-4)
    assert options == ({"ftol": 1e-12} if method == "SLSQP" else {"gtol": 1e-10})


@pytest.mark.scipy
def test_native_iteration_limit_is_reported_as_unsuccessful_result():
    from diffractix.solver.optimizers.scipy import solve_scipy
    result = solve_scipy(quadratic_problem(), method="SLSQP", options={"maxiter": 1})
    assert not result.success
    assert "iteration" in result.message.lower()
    assert np.all(np.isfinite(result.x))
    assert np.isfinite(result.cost)
