from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

import numpy as np

from .scale import Scaling


@dataclass(frozen=True)
class Problem:
    """Compiled numerical optimization problem."""

    x0: np.ndarray
    x_lower: np.ndarray
    x_upper: np.ndarray

    objective: Callable
    gradient: Callable
    objective_hessian: Callable

    constraints: Callable
    jacobian: Callable
    constraint_hessian: Callable # TODO allow for sparse hessians and jacobians
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray

    # Always initialized after construction; callers can pass an existing
    # Scaling when constructing a transformed Problem.
    scaling: Scaling = None  # type: ignore[assignment]
    proximal: np.ndarray | None = None  # Physical coordinates, even after scaling.
    proximal_weight: float = 1.0

    def __post_init__(self):
        if len(self.x0) != len(self.x_lower) or len(self.x0) != len(self.x_upper):
            raise ValueError("Variable bounds must match x0.")
        if len(self.constraint_lower) != len(self.constraint_upper):
            raise ValueError("Constraint bounds must have the same length.")

        weight = self.proximal_weight
        if not isinstance(weight, Real) or isinstance(weight, bool):
            raise TypeError("proximal_weight must be a real numeric scalar.")
        if not isfinite(weight) or weight < 0:
            raise ValueError("proximal_weight must be finite and non-negative.")
        if self.proximal is not None:
            if weight == 0:
                raise ValueError("proximal_weight must be positive when proximal is supplied.")
            reference = np.asarray(self.proximal)
            if reference.shape != (self.variable_count,):
                raise ValueError(f"proximal must have shape ({self.variable_count},), got {reference.shape}.")
            if reference.dtype.kind not in "iuf" or not np.all(np.isfinite(reference)):
                raise ValueError("proximal must contain only finite real values.")
            if self.scaling is None:
                reference = reference.copy()
            object.__setattr__(self, "proximal", reference)

        if self.scaling is None:
            object.__setattr__(self, "scaling", Scaling(self))

    @property
    def variable_count(self) -> int:
        return len(self.x0)

    @property
    def constraint_count(self) -> int:
        return len(self.constraint_lower)

    def scale(self) -> Problem:
        return self.scaling.scale(self)

    def unscale(self) -> Problem:
        return self.scaling.unscale(self)

    def scaled_proximal(self):
        """Return the scaled reference and weights, with pins excluded."""
        if self.proximal is None:
            return None
        free = self.x_lower != self.x_upper
        reference = np.zeros(self.variable_count)
        with np.errstate(over="ignore", invalid="ignore"):
            reference[free] = (
                (self.proximal[free] - self.scaling.x_center[free]) / self.scaling.x_scale[free]
            )
        if not np.all(np.isfinite(reference)):
            raise ValueError("proximal must be finite in the solver's scaled coordinates.")
        return reference, self.proximal_weight * free

    def proximal_cost(self, x) -> float:
        """Evaluate the movement penalty at a point in physical coordinates."""
        proximal = self.scaled_proximal()
        if proximal is None:
            return 0.0
        reference, weights = proximal
        free = weights != 0
        scaled_x = (x[free] - self.scaling.x_center[free]) / self.scaling.x_scale[free]
        return float(weights[free] @ (scaled_x - reference[free])**2)

    def with_proximal(self) -> Problem:
        """Fold the penalty into the callbacks of a scaled optimizer problem."""
        proximal = self.scaled_proximal()
        if proximal is None:
            return self
        reference, weights = proximal
        hessian = np.diag(2 * weights)

        def objective(y):
            return self.objective(y) + weights @ (y - reference)**2

        def gradient(y):
            return self.gradient(y) + 2 * weights * (y - reference)

        def objective_hessian(y):
            return self.objective_hessian(y) + hessian

        return replace(
            self, objective=objective, gradient=gradient, objective_hessian=objective_hessian,
            proximal=None,
        )
