from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """Soft optimization objective producing one or more residuals."""

    evaluate: Callable
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self):
        if not callable(self.evaluate):
            raise TypeError("evaluate must be callable.")
        if self.weight < 0:
            raise ValueError("weight must be >= 0.")

    def __call__(self, *args):
        return self.evaluate(*args)