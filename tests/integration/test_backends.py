"""Real optimization backends; optional packages skip per test."""

from __future__ import annotations

import importlib
from dataclasses import replace
import numpy as np
import pytest
from tests.helpers.optimization import constrained_problem, quadratic_problem


@pytest.mark.scipy
def test_variable_constraint_scaling_scipy_constrained_quadratic():
    from diffractix.solver.backends.scipy import solve_scipy

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
    from diffractix.solver.backends.scipy import solve_scipy

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
        importlib.import_module("diffractix.solver.backends.nlopt")
    )

    result = backend.solve_nlopt(
        constrained_problem(),
        method="LD_SLSQP",
        options={"xtol_rel": 1e-9},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 1.0], atol=1e-5)
    assert result.cost == pytest.approx(0.0, abs=1e-10)


@pytest.mark.ipopt
def test_real_ipopt_backend_smoke():
    pytest.importorskip("cyipopt")
    backend = importlib.reload(
        importlib.import_module("diffractix.solver.backends.ipopt")
    )

    result = backend.solve_ipopt(
        constrained_problem(),
        options={"print_level": 0},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 1.0], atol=1e-5)
    assert result.cost == pytest.approx(0.0, abs=1e-10)
