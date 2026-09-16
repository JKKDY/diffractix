from __future__ import annotations

from ..problem import Problem


def solve_nlopt(
    problem: Problem,
    method: str | None = None,
    options: dict | None = None,
):
    """Solve a compiled optimization problem with NLopt.

    Import the optional NLopt binding inside this function when implementing
    the adapter so importing :mod:`diffractix.solver.backends` stays safe.
    """
    raise NotImplementedError("The NLopt solver backend is not implemented.")
