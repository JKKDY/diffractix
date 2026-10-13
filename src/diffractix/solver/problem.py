from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import autograd.numpy as np

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
    constraint_hessian: Callable
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray

    # Always initialized after construction; callers can pass an existing
    # Scaling when constructing a transformed Problem.
    scaling: Scaling = None  # type: ignore[assignment]

    def __post_init__(self):
        if len(self.x0) != len(self.x_lower) or len(self.x0) != len(self.x_upper):
            raise ValueError("Variable bounds must match x0.")
        if len(self.constraint_lower) != len(self.constraint_upper):
            raise ValueError("Constraint bounds must have the same length.")

        if self.scaling is None:
            object.__setattr__(self, "scaling", Scaling(self))

    @property
    def n_variables(self) -> int:
        return len(self.x0)

    @property
    def n_constraints(self) -> int:
        return len(self.constraint_lower)

    def scale(self) -> Problem:
        return self.scaling.scale(self)

    def unscale(self) -> Problem:
        return self.scaling.unscale(self)
