"""A compiled CasADi NLP with numerical bounds and scaling supplied per solve."""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from diffractix.backend._casadi import _output, _symbol, ca
from diffractix.solver.solution import SolverResult


def _compile_nlp(objective, constraints, variable_count, method, options):
    if not ca.has_nlpsol(method):
        raise ValueError(f"CasADi nlpsol plugin {method!r} is unavailable in this installation.")

    settings = {"print_time": False, "error_on_fail": False}
    if method == "sqpmethod":
        settings.update(
            qpsol="qrqp", print_header=False, print_iteration=False, print_status=False,
            tol_pr=1e-10, tol_du=1e-8,
            convexify_strategy="eigen-clip", max_iter_ls=12,
        )
        
    settings.update(options)
    if method == "sqpmethod" and settings["qpsol"] == "qrqp":
        settings["qpsol_options"] = {
            "print_header": False,
            "print_iter": False,
            "error_on_fail": False,
            **settings.get("qpsol_options", {}),
        }

    (x,), theta = _symbol(0, (variable_count,), np.dtype(float))
    ((cost,), _, _) = _output(objective(theta))
    ((g,), _, _) = _output(constraints(theta))
    constraint_count = g.numel()

    y = ca.SX.sym("y", variable_count)
    center = ca.SX.sym("center", variable_count)
    scale = ca.SX.sym("scale", variable_count)
    constraint_scale = ca.SX.sym("constraint_scale", constraint_count)
    physical = center + scale * y
    nlp = {
        "x": y,
        "p": ca.vertcat(center, scale, constraint_scale),
        "f": ca.substitute(cost, x, physical),
        # CasADi 3.7.2 SQP can corrupt memory on some zero-row NLPs.
        "g": (ca.substitute(g, x, physical) / constraint_scale
              if constraint_count else ca.SX.zeros(1)),
    }
    return ca.nlpsol("diffractix", method, nlp, settings), constraint_count


def solve_casadi(problem, method=None, options=None, *, cache, objective, constraints, active_rows):
    """Solve using a compiled NLP shared across calls with the same options."""
    method = "sqpmethod" if method is None else method
    options = {} if options is None else options
    entry = cache.get(solve_casadi)
    if entry is None or entry[:2] != (method, options):
        compiled, constraint_count = _compile_nlp(objective, constraints, problem.variable_count, method, options)
        entry = cache[solve_casadi] = (method, deepcopy(options), compiled, constraint_count)
    _, _, compiled, constraint_count = entry
    lower = np.full(max(1, constraint_count), -np.inf)
    upper = np.full(max(1, constraint_count), np.inf)
    constraint_scale = np.ones(constraint_count)
    lower[active_rows] = problem.constraint_lower
    upper[active_rows] = problem.constraint_upper
    constraint_scale[active_rows] = problem.scaling.constraint_scale
    parameters = np.concatenate((
        problem.scaling.x_center, problem.scaling.x_scale, constraint_scale,
    ))

    try:
        result = compiled(
            x0=problem.x0, lbx=problem.x_lower, ubx=problem.x_upper,
            lbg=lower, ubg=upper, p=parameters,
        )
    except RuntimeError as error:
        return SolverResult(
            x=problem.x0, success=False, cost=float(problem.objective(problem.x0)),
            message=f"CasADi {method}: {str(error).splitlines()[-1]}", scaling=problem.scaling,
        )

    try:
        stats = compiled.stats()
        message = str(stats.get("return_status", "No status returned"))
    except RuntimeError as error:
        stats = {}
        message = f"Status unavailable ({str(error).splitlines()[-1]})"
    x = result["x"].full().ravel()
    cost = float(result["f"])
    success = bool(stats.get("success", False))
    if not np.all(np.isfinite(x)):
        x = problem.x0
        cost = float(problem.objective(x))
        success = False
        message = f"{message} (non-finite iterate)"
    if not np.isfinite(cost):
        success = False
        message = f"{message} (non-finite objective)"
    return SolverResult(
        x=x, success=success, cost=cost, message=f"CasADi {method}: {message}",
        iterations=stats.get("iter_count"), function_evaluations=stats.get("n_call_nlp_fg"),
        raw=stats, scaling=problem.scaling,
    )
