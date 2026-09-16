from __future__ import annotations

from ..problem import Problem


def solve_ipopt(problem: Problem, method: str | None = None):
    """Solve a compiled optimization problem with IPOPT.

    Import the optional IPOPT binding inside this function when implementing
    the adapter so importing :mod:`diffractix.solver.backends` stays safe.
    """
    raise NotImplementedError("The IPOPT solver backend is not implemented.")
