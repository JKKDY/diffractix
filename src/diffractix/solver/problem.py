from dataclasses import dataclass
from collections.abc import Callable

import autograd.numpy as np


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

    def __post_init__(self):
        if len(self.x0) != len(self.x_lower) or len(self.x0) != len(self.x_upper):
            raise ValueError("Variable bounds must match x0.")
        if len(self.constraint_lower) != len(self.constraint_upper):
            raise ValueError("Constraint bounds must have the same length.")

    @property
    def n_variables(self):
        return len(self.x0)

    @property
    def n_constraints(self):
        return len(self.constraint_lower)
