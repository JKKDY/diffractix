from __future__ import annotations

import numpy as np
import cyipopt

from ..problem import Problem
from ..result import OptimizationResult


class _IpoptProblem:
    def __init__(self, problem: Problem):
        self.problem = problem

    def objective(self, x):
        return self.problem.objective(x)

    def gradient(self, x):
        return np.asarray(self.problem.gradient(x))

    def constraints(self, x):
        return np.asarray(self.problem.constraints(x))

    def jacobian(self, x):
        return np.asarray(self.problem.jacobian(x)).ravel()


def solve_ipopt(problem: Problem, method: str | None = None) -> OptimizationResult:
    """Solve a compiled optimization problem with IPOPT."""

    if method is not None:
        raise ValueError("IPOPT does not expose alternative optimization methods.")

    callbacks = _IpoptProblem(problem)

    solver = cyipopt.Problem(
        n=problem.n_variables,
        m=problem.n_constraints,
        problem_obj=callbacks,
        lb=problem.x_lower,
        ub=problem.x_upper,
        cl=problem.constraint_lower,
        cu=problem.constraint_upper,
    )

    solver.add_option(
        "hessian_approximation",
        "limited-memory",
    )

    x, info = solver.solve(problem.x0)

    message = info.get("status_msg", "")
    if isinstance(message, bytes):
        message = message.decode()

    return OptimizationResult(
        x=np.asarray(x),
        success=info.get("status") == 0,
        cost=float(info["obj_val"]),
        message=str(message),
        raw=info,
    )