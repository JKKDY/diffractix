from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect

from diffractix.graph import Node


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


def _callable_arity(function):
    signature = inspect.signature(function)

    try:
        signature.bind()
        accepts_zero = True
    except TypeError:
        accepts_zero = False

    try:
        signature.bind(object())
        accepts_one = True
    except TypeError:
        accepts_one = False

    if accepts_zero == accepts_one:
        raise TypeError("Callable must accept exactly zero or one argument.")

    return 0 if accepts_zero else 1


def normalize_to_objective(value) -> Objective:
    if isinstance(value, Node):
        return Objective(value)

    if isinstance(value, Objective):
        objective = value
    elif callable(value):
        objective = Objective(value)
    else:
        raise TypeError("Objective must be a Node, Objective, or callable.")

    if isinstance(objective.evaluate, Node):
        return objective

    if _callable_arity(objective.evaluate) == 1:
        return objective

    result = objective.evaluate()

    if not isinstance(result, Node):
        raise TypeError("A zero-argument objective callable must return a Node.")

    return Objective(
        evaluate=result,
        weight=objective.weight,
        label=objective.label,
    )