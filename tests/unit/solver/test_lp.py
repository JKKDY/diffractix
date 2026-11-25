import sys

import numpy as np
import pytest

from diffractix.solver import lp


@pytest.mark.parametrize("use_scipy", [True, False])
@pytest.mark.parametrize("cost, options, status, optimum", [
    pytest.param([0], {"A_ub": [[1], [-1]], "b_ub": [-1, 2], "free": True},
                 "optimal", 0, id="free-negative-variable"),
    pytest.param([1, 2], {"A_eq": [[1, 1], [2, 2], [0, 0]], "b_eq": [1, 2, 0]},
                 "optimal", 1, id="redundant-equalities"),
    pytest.param([-1], {"A_ub": [[1], [2], [3]], "b_ub": [1, 2, 4]},
                 "optimal", -1, id="parallel-inequalities"),
    pytest.param([0], {"A_ub": [[1], [-2]], "b_ub": [1, -4], "free": True},
                 "infeasible", None, id="parallel-conflict"),
    pytest.param([0, 0], {"A_eq": [[1, 1], [2, 2]], "b_eq": [1, 3]},
                 "infeasible", None, id="equality-conflict"),
    pytest.param([-1], {"A_ub": [[-1]], "b_ub": [-1]},
                 "unbounded", None, id="unbounded-after-phase-one"),
    pytest.param([-10, 57, 9, 24], {
        "A_ub": [[0.5, -5.5, -2.5, 9], [0.5, -1.5, -0.5, 1], [1, 0, 0, 0]],
        "b_ub": [0, 0, 1],
    }, "optimal", -1, id="degenerate-cycling-example"),
])
def test_lp_status_and_solution(monkeypatch, use_scipy, cost, options, status, optimum):
    if use_scipy:
        pytest.importorskip("scipy")
    else:
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)

    result = lp.solve_lp(cost, **options)
    assert result.status == status
    if optimum is None:
        assert result.x is None
    else:
        assert np.dot(cost, result.x) == pytest.approx(optimum)
        if "A_ub" in options:
            assert np.all(np.array(options["A_ub"]) @ result.x <= np.array(options["b_ub"]) + 1e-10)
        if "A_eq" in options:
            np.testing.assert_allclose(np.array(options["A_eq"]) @ result.x, options["b_eq"], atol=1e-10)


def test_numpy_lp_returns_failed_when_it_cannot_pivot_reliably(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    assert lp.solve_lp([-1], A_ub=[[1e-20]], b_ub=[1]).status == "failed"
    monkeypatch.setattr(lp, "_MAX_PIVOTS", 0)
    assert lp.solve_lp([-1], A_ub=[[1]], b_ub=[1]).status == "failed"
