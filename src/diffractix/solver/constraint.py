from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import autograd.numpy as np


@dataclass(frozen=True)
class Constraint:
    """Hard nonlinear constraint evaluated between lower and upper bounds."""

    evaluate: Callable
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    label: str | None = None

    def __post_init__(self):
        if not callable(self.evaluate):
            raise TypeError("evaluate must be callable.")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound.")

    def __call__(self, *args):
        return self.evaluate(*args)