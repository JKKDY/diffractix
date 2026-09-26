from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from ..problem import Problem
from ..result import OptimizationResult

DEFAULT_METHOD = "trust-constr"
SUPPORTED_METHODS = {"trust-constr", "SLSQP"}


def solve_scipy(
    problem: Problem,
    method: str | None = None,
    options: dict | None = None,
) -> OptimizationResult:
    """Solve a compiled optimization problem with SciPy."""
    method = DEFAULT_METHOD if method is None else method
    options = {} if options is None else dict(options)
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported SciPy method {method!r}. Choose one of {sorted(SUPPORTED_METHODS)!r}."
        )

    constraints = []
    if problem.n_constraints:
        equality = problem.constraint_lower == problem.constraint_upper

        for indices in (
            np.flatnonzero(equality),
            np.flatnonzero(~equality),
        ):
            if not len(indices):
                continue

            def values(theta, indices=indices):
                return problem.constraints(theta)[indices]

            def jacobian(theta, indices=indices):
                return problem.jacobian(theta)[indices]

            constraint_kwargs = {"jac": jacobian}
            if method == "trust-constr":
                def hessian(theta, multipliers, indices=indices):
                    full_multipliers = np.zeros(problem.n_constraints)
                    full_multipliers[indices] = multipliers
                    return problem.constraint_hessian(theta, full_multipliers)

                constraint_kwargs["hess"] = hessian

            constraints.append(
                NonlinearConstraint(
                    values,
                    problem.constraint_lower[indices],
                    problem.constraint_upper[indices],
                    **constraint_kwargs,
                )
            )

    kwargs = {}
    if method == "trust-constr":
        kwargs["hess"] = problem.objective_hessian

    res = minimize(
        problem.objective,
        problem.x0,
        method=method,
        jac=problem.gradient,
        bounds=Bounds(problem.x_lower, problem.x_upper),
        constraints=tuple(constraints),
        options=options,
        **kwargs,
    )

    return OptimizationResult(
        x=res.x,
        success=bool(res.success),
        cost=float(res.fun),
        message=str(res.message),
        iterations=getattr(res, "nit", None),
        raw=res,
    )
