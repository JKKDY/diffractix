from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect

from diffractix.graph import Node
from .utils import callable_arity


@dataclass(frozen=True)
class Objective:
    """Soft optimization objective producing one or more residuals."""
    evaluate: Node | Callable
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.evaluate, Node) and not callable(self.evaluate):
            raise TypeError("evaluate must be a Node or callable.")
        if self.weight < 0:
            raise ValueError("weight must be >= 0.")



def normalize_to_objective(value) -> Objective:
    if isinstance(value, Node):
        return Objective(value)

    objective = value if isinstance(value, Objective) else Objective(value)
    if isinstance(objective.evaluate, Node) or callable_arity(objective.evaluate) == 1:
        return objective

    result = objective.evaluate()
    if not isinstance(result, Node):
        raise TypeError("A zero-argument objective callable must return a Node.")
    return Objective(result, objective.weight, objective.label)