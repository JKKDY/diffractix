from __future__ import annotations

import numpy as np
import cyipopt

from ..problem import Problem
from ..solution import OptimizationResult


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


def solve_ipopt(problem: Problem, method: str | None = None, options: dict | None = None) -> OptimizationResult:
    """Solve a compiled optimization problem with IPOPT."""

    if method is not None:
        raise ValueError("IPOPT does not expose alternative optimization methods.")

    options = {} if options is None else dict(options)

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

    ipopt_options = {
        "hessian_approximation": "limited-memory",
        **options,
    }
    for name, value in ipopt_options.items():
        solver.add_option(name, value)

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
