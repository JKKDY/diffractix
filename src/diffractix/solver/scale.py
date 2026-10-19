from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .problem import Problem
    from .result import OptimizationResult


@dataclass(frozen=True, init=False)
class Scaling:
    """Affine variable and constraint scaling for numerical optimization.

    Physical variables x = x_center + x_scale * y.
    Scaled constraints h(y) = (g(x) - constraint_center) / constraint_scale.

    The objective is not rescaled.
    """

    x_center: np.ndarray
    x_scale: np.ndarray
    constraint_center: np.ndarray
    constraint_scale: np.ndarray

    def __init__(self, problem: Problem):
        x0 = np.asarray(problem.x0, dtype=float)
        lower = np.asarray(problem.x_lower, dtype=float)
        upper = np.asarray(problem.x_upper, dtype=float)

        scales = np.empty(len(x0), dtype=float)
        for i, (value, lo, hi) in enumerate(zip(x0, lower, upper)):
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                # Half-width without computing hi - lo (which could overflow).
                scale = 0.5 * hi - 0.5 * lo
            else:
                scale = abs(value) if np.isfinite(value) and value != 0 else 1.0

            if not np.isfinite(scale) or scale <= 0:
                scale = abs(value) if np.isfinite(value) and value != 0 else 1.0

            scales[i] = scale

        # Measure each constraint's sensitivity in normalized variable units.
        # Avoid evaluating a Jacobian when there are no constraints.
        if problem.n_constraints and len(x0):
            jac = np.asarray(problem.jacobian(x0), dtype=float)
            if jac.shape != (problem.n_constraints, len(x0)):
                raise ValueError("Constraint Jacobian has incorrect shape.")
            constraint_scales = np.max(np.abs(jac) * scales[None, :], axis=1)
            constraint_scales = np.where(
                np.isfinite(constraint_scales) & (constraint_scales > 0),
                constraint_scales,
                1.0,
            )
        else:
            constraint_scales = np.ones(problem.n_constraints)

        object.__setattr__(self, "x_center", x0.copy())
        object.__setattr__(self, "x_scale", scales)
        object.__setattr__(self, "constraint_center", np.zeros(problem.n_constraints))
        object.__setattr__(self, "constraint_scale", constraint_scales)

        for values in (
            self.x_center,
            self.x_scale,
            self.constraint_center,
            self.constraint_scale,
        ):
            values.setflags(write=False)

    def scale(self, obj: Problem | OptimizationResult) -> Problem | OptimizationResult:
        """Return an object expressed in backend numerical coordinates."""
        from .problem import Problem
        from .result import OptimizationResult

        c, s = self.x_center, self.x_scale
        a, t = self.constraint_center, self.constraint_scale

        if isinstance(obj, OptimizationResult):
            return replace(obj, x=(obj.x - c) / s, scaling=self)

        if isinstance(obj, Problem):
            def physical(y):
                return c + s * y

            def objective(y):
                return obj.objective(physical(y))

            def gradient(y):
                return s * obj.gradient(physical(y))

            def objective_hessian(y):
                return s[:, None] * obj.objective_hessian(physical(y)) * s[None, :]

            def constraints(y):
                return (obj.constraints(physical(y)) - a) / t

            def jacobian(y):
                return obj.jacobian(physical(y)) * s[None, :] / t[:, None]

            def constraint_hessian(y, multipliers):
                return (
                    s[:, None]
                    * obj.constraint_hessian(physical(y), multipliers / t)
                    * s[None, :]
                )

            return Problem(
                x0=(obj.x0 - c) / s,
                x_lower=(obj.x_lower - c) / s,
                x_upper=(obj.x_upper - c) / s,
                objective=objective,
                gradient=gradient,
                objective_hessian=objective_hessian,
                constraints=constraints,
                jacobian=jacobian,
                constraint_hessian=constraint_hessian,
                constraint_lower=(obj.constraint_lower - a) / t,
                constraint_upper=(obj.constraint_upper - a) / t,
                scaling=self,
            )

        raise TypeError(f"Cannot scale {type(obj).__name__}.")

    def unscale(self, obj: Problem | OptimizationResult) -> Problem | OptimizationResult:
        """Return an object expressed in physical SI coordinates."""
        from .problem import Problem
        from .result import OptimizationResult

        c, s = self.x_center, self.x_scale
        a, t = self.constraint_center, self.constraint_scale

        if isinstance(obj, OptimizationResult):
            # Cost and raw backend diagnostics remain exactly as reported.
            return replace(obj, x=c + s * obj.x, scaling=self)

        if isinstance(obj, Problem):
            def numerical(x):
                return (x - c) / s

            def objective(x):
                return obj.objective(numerical(x))

            def gradient(x):
                return obj.gradient(numerical(x)) / s

            def objective_hessian(x):
                return obj.objective_hessian(numerical(x)) / s[:, None] / s[None, :]

            def constraints(x):
                return a + t * obj.constraints(numerical(x))

            def jacobian(x):
                return obj.jacobian(numerical(x)) * t[:, None] / s[None, :]

            def constraint_hessian(x, multipliers):
                return (
                    obj.constraint_hessian(numerical(x), multipliers * t)
                    / s[:, None]
                    / s[None, :]
                )

            return Problem(
                x0=c + s * obj.x0,
                x_lower=c + s * obj.x_lower,
                x_upper=c + s * obj.x_upper,
                objective=objective,
                gradient=gradient,
                objective_hessian=objective_hessian,
                constraints=constraints,
                jacobian=jacobian,
                constraint_hessian=constraint_hessian,
                constraint_lower=a + t * obj.constraint_lower,
                constraint_upper=a + t * obj.constraint_upper,
                scaling=self,
            )

        raise TypeError(f"Cannot unscale {type(obj).__name__}.")
