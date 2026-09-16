from __future__ import annotations

import numpy as np
import nlopt

from ..problem import Problem
from ..result import OptimizationResult


DEFAULT_METHOD = "LD_SLSQP"

_OPTION_SETTERS = {
    "stopval": "set_stopval",
    "ftol_rel": "set_ftol_rel",
    "ftol_abs": "set_ftol_abs",
    "xtol_rel": "set_xtol_rel",
    "xtol_abs": "set_xtol_abs",
    "x_weights": "set_x_weights",
    "maxeval": "set_maxeval",
    "maxtime": "set_maxtime",
    "initial_step": "set_initial_step",
    "population": "set_population",
    "vector_storage": "set_vector_storage",
}

_STATUS_MESSAGES = {
    nlopt.SUCCESS: "Success.",
    nlopt.STOPVAL_REACHED: "Stop value reached.",
    nlopt.FTOL_REACHED: "Function tolerance reached.",
    nlopt.XTOL_REACHED: "Parameter tolerance reached.",
    nlopt.MAXEVAL_REACHED: "Maximum evaluations reached.",
    nlopt.MAXTIME_REACHED: "Maximum time reached.",
}


def _algorithm(method):
    method = DEFAULT_METHOD if method is None else method
    method = method.removeprefix("NLOPT_")

    algorithm = getattr(nlopt, method, None)

    if not isinstance(algorithm, int):
        raise ValueError(f"Unknown NLopt method {method!r}.")

    return algorithm


def _apply_options(optimizer, options):
    for name, value in options.items():
        setter = _OPTION_SETTERS.get(name)

        if setter is not None:
            getattr(optimizer, setter)(value)
        elif optimizer.has_param(name):
            optimizer.set_param(name, value)
        else:
            raise ValueError(f"Unknown NLopt option {name!r}.")


def solve_nlopt(
    problem: Problem,
    method: str | None = None,
    options: dict | None = None,
) -> OptimizationResult:
    """Solve a compiled optimization problem with NLopt."""

    options = {} if options is None else dict(options)

    constraint_tol = options.pop("constraint_tol", 1e-8)

    optimizer = nlopt.opt(
        _algorithm(method),
        problem.n_variables,
    )

    optimizer.set_lower_bounds(problem.x_lower)
    optimizer.set_upper_bounds(problem.x_upper)

    def objective(x, grad):
        if grad.size:
            grad[:] = np.asarray(problem.gradient(x))

        return float(problem.objective(x))

    optimizer.set_min_objective(objective)

    lower = np.asarray(problem.constraint_lower)
    upper = np.asarray(problem.constraint_upper)

    equality = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (lower == upper)
    )

    upper_indices = np.flatnonzero(
        np.isfinite(upper) & ~equality
    )
    lower_indices = np.flatnonzero(
        np.isfinite(lower) & ~equality
    )
    equality_indices = np.flatnonzero(equality)

    if upper_indices.size or lower_indices.size:
        indices = np.concatenate([
            upper_indices,
            lower_indices,
        ])

        signs = np.concatenate([
            np.ones(upper_indices.size),
            -np.ones(lower_indices.size),
        ])

        offsets = np.concatenate([
            -upper[upper_indices],
            lower[lower_indices],
        ])

        def inequalities(result, x, grad):
            values = np.asarray(problem.constraints(x))

            result[:] = (
                signs * values[indices]
                + offsets
            )

            if grad.size:
                jacobian = np.asarray(problem.jacobian(x))
                grad[:] = (
                    signs[:, None]
                    * jacobian[indices]
                )

        optimizer.add_inequality_mconstraint(
            inequalities,
            np.full(indices.size, constraint_tol),
        )

    if equality_indices.size:

        def equalities(result, x, grad):
            values = np.asarray(problem.constraints(x))

            result[:] = (
                values[equality_indices]
                - lower[equality_indices]
            )

            if grad.size:
                jacobian = np.asarray(problem.jacobian(x))
                grad[:] = jacobian[equality_indices]

        optimizer.add_equality_mconstraint(
            equalities,
            np.full(
                equality_indices.size,
                constraint_tol,
            ),
        )

    _apply_options(optimizer, options)

    x = optimizer.optimize(problem.x0)

    status = optimizer.last_optimize_result()

    return OptimizationResult(
        x=np.asarray(x),
        success=status in (
            nlopt.SUCCESS,
            nlopt.STOPVAL_REACHED,
            nlopt.FTOL_REACHED,
            nlopt.XTOL_REACHED,
        ),
        cost=float(optimizer.last_optimum_value()),
        message=_STATUS_MESSAGES.get(
            status,
            f"NLopt terminated with status {status}.",
        ),
        raw={
            "status": status,
            "algorithm": optimizer.get_algorithm_name(),
            "evaluations": optimizer.get_numevals(),
        },
    )