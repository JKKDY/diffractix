from __future__ import annotations

from scipy.optimize import Bounds, NonlinearConstraint, minimize

from ..problem import Problem
from ..result import OptimizationResult

DEFAULT_METHOD = "trust-constr"
SUPPORTED_METHODS = {"trust-constr", "SLSQP"}


def solve_scipy(problem: Problem, method: str | None = None) -> OptimizationResult:
    """Solve a compiled optimization problem with SciPy."""
    method = method or DEFAULT_METHOD
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported SciPy method {method!r}. Choose one of {sorted(SUPPORTED_METHODS)!r}."
        )

    constraints = ()
    if problem.n_constraints:
        constraints = (
            NonlinearConstraint(
                problem.constraints,
                problem.constraint_lower,
                problem.constraint_upper,
                jac=problem.jacobian,
            ),
        )

    res = minimize(
        problem.objective,
        problem.x0,
        method=method,
        jac=problem.gradient,
        bounds=Bounds(problem.x_lower, problem.x_upper),
        constraints=constraints,
    )

    return OptimizationResult(
        x=res.x,
        success=bool(res.success),
        cost=float(res.fun),
        message=str(res.message),
        iterations=getattr(res, "nit", None),
        raw=res,
    )