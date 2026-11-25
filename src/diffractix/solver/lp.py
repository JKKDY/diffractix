"""Solve small certificate LPs with HiGHS or a NumPy fallback."""

from typing import Literal, NamedTuple

import numpy as np


class LPResult(NamedTuple):
    status: Literal["optimal", "infeasible", "unbounded", "failed"]
    x: np.ndarray | None = None


def solve_lp(c, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None, free=False):
    """Minimize c @ x with linear inequalities or equalities and uniform bounds."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                return _solve_numpy(c, A_ub, b_ub, A_eq, b_eq, free)
            except (ValueError, OverflowError, FloatingPointError):
                return LPResult("failed")

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(None, None) if free else (0, None), method="highs",
    )
    status = {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(result.status, "failed")
    return LPResult(status, result.x if status == "optimal" else None)


_TOL = 1e-10
_MAX_PIVOTS = 1000


def _pivot(tableau, basis, row, column):
    coefficients = tableau[:, column].copy()
    coefficients[row] = 0
    tableau[row] /= tableau[row, column]
    tableau -= coefficients[:, None] * tableau[row]
    tableau[:, column] = 0
    tableau[row, column] = 1
    basis[row] = column


def _simplex(tableau, basis, cost):
    for _ in range(_MAX_PIVOTS):
        rhs = tableau[:, -1]
        if np.any(rhs < -_TOL):
            return "failed"
        reduced_cost = cost - cost[basis] @ tableau[:, :-1]
        reduced_cost[basis] = 0
        entering = np.flatnonzero(reduced_cost < -_TOL)
        if not entering.size:
            return "optimal"

        # Bland's rule chooses the lowest-index entering and tied leaving variables.
        column = entering[0]
        coefficients = tableau[:, column]
        candidates = np.flatnonzero(coefficients > _TOL)
        if not candidates.size:
            return "failed" if np.any(coefficients > 0) else "unbounded"
        ratios = np.maximum(rhs[candidates], 0) / coefficients[candidates]
        tied = candidates[ratios == ratios.min()]
        row = tied[np.argmin(basis[tied])]
        _pivot(tableau, basis, row, column)
    return "failed"


def _solve_numpy(c, A_ub, b_ub, A_eq, b_eq, free):
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or not np.all(np.isfinite(c)):
        return LPResult("failed")
    n = c.size
    inequalities = np.empty((0, n)) if A_ub is None else np.asarray(A_ub, dtype=float)
    equalities = np.empty((0, n)) if A_eq is None else np.asarray(A_eq, dtype=float)
    m_ub = len(inequalities)
    matrix = np.vstack((inequalities, equalities))
    rhs = np.concatenate(([] if b_ub is None else b_ub, [] if b_eq is None else b_eq)).astype(float)
    if matrix.shape != (rhs.size, n) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        return LPResult("failed")

    # Normalize the rows before introducing slacks. A free variable is x+ - x-.
    scale = np.maximum(np.max(np.abs(matrix), axis=1, initial=0), np.abs(rhs))
    scale[scale == 0] = 1
    matrix /= scale[:, None]
    rhs /= scale
    if free:
        matrix = np.hstack((matrix, -matrix))
        c = np.concatenate((c, -c))

    # Rows with negative RHS or equality constraints need an artificial basic variable.
    signs = np.where(rhs < 0, -1, 1)
    artificial_rows = np.flatnonzero((rhs < 0) | (np.arange(rhs.size) >= m_ub))
    slack_rows = np.arange(m_ub)
    nonartificial = c.size + m_ub
    columns = nonartificial + artificial_rows.size
    tableau = np.zeros((rhs.size, columns + 1))
    tableau[:, :c.size] = matrix * signs[:, None]
    tableau[:, -1] = np.abs(rhs)
    tableau[slack_rows, c.size + slack_rows] = signs[slack_rows]
    basis = c.size + np.arange(rhs.size)
    basis[artificial_rows] = np.arange(nonartificial, columns)
    tableau[artificial_rows, basis[artificial_rows]] = 1

    if artificial_rows.size:
        phase_one_cost = np.zeros(columns)
        phase_one_cost[nonartificial:] = 1
        if _simplex(tableau, basis, phase_one_cost) != "optimal":
            return LPResult("failed")
        if tableau[basis >= nonartificial, -1].sum() > _TOL:
            return LPResult("infeasible")

        # Pivot out zero artificial variables, or remove their redundant rows.
        keep = np.ones(rhs.size, dtype=bool)
        for row in np.flatnonzero(basis >= nonartificial):
            candidates = np.flatnonzero(np.abs(tableau[row, :nonartificial]) > _TOL)
            if candidates.size:
                if abs(tableau[row, -1]) > _TOL:
                    return LPResult("failed")
                tableau[row, -1] = 0
                _pivot(tableau, basis, row, candidates[0])
            else:
                keep[row] = False
        tableau = np.column_stack((tableau[keep, :nonartificial], tableau[keep, -1]))
        basis = basis[keep]

    cost = np.zeros(nonartificial)
    cost[:c.size] = c / (np.max(np.abs(c), initial=0) or 1)
    status = _simplex(tableau, basis, cost)
    if status != "optimal":
        return LPResult(status)
    solution = np.zeros(nonartificial)
    solution[basis] = np.maximum(tableau[:, -1], 0)
    return LPResult("optimal", solution[:n] - solution[n:2*n] if free else solution[:n])
