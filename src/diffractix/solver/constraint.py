from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import autograd.numpy as np

from diffractix.graph import Comparison, Literal, Node, Relation

from .utils import callable_arity


@dataclass(frozen=True)
class Constraint:
    """Hard nonlinear constraint evaluated between lower and upper bounds."""
    evaluate: Node | Callable
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.evaluate, Node) and not callable(self.evaluate):
            raise TypeError("evaluate must be a Node or callable.")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound.")


def _from_comparison(comparison) -> Constraint:
    lhs, rhs, rel = comparison.left, comparison.right, comparison.relation

    if isinstance(rhs, Literal):
        if rel is Relation.LE: return Constraint(lhs, upper_bound=rhs.value)
        if rel is Relation.GE: return Constraint(lhs, lower_bound=rhs.value)
        if rel is Relation.EQ: return Constraint(lhs, rhs.value, rhs.value)

    if isinstance(lhs, Literal):
        if rel is Relation.LE: return Constraint(rhs, lower_bound=lhs.value)
        if rel is Relation.GE: return Constraint(rhs, upper_bound=lhs.value)
        if rel is Relation.EQ: return Constraint(rhs, lhs.value, lhs.value)

    residual = lhs - rhs
    if rel is Relation.LE: return Constraint(residual, upper_bound=0.0)
    if rel is Relation.GE: return Constraint(residual, lower_bound=0.0)
    if rel is Relation.EQ: return Constraint(residual, 0.0, 0.0)
    raise ValueError(f"Unsupported relation {rel!r}.")


def normalize_to_constraint(value) -> Constraint:
    if isinstance(value, Comparison):
        return _from_comparison(value)

    if not isinstance(value, Constraint):
        if not callable(value) or callable_arity(value) != 0:
            raise TypeError(
                "Constraint must be a Comparison, Constraint, "
                "or zero-argument callable returning a Comparison."
            )
        result = value()
        if not isinstance(result, Comparison):
            raise TypeError("A zero-argument constraint callable must return a Comparison.")
        return _from_comparison(result)

    if isinstance(value.evaluate, Node) or callable_arity(value.evaluate) == 1:
        return value

    result = value.evaluate()
    if not isinstance(result, Node):
        raise TypeError("A zero-argument Constraint callable must return a Node.")
    return Constraint(result, value.lower_bound, value.upper_bound, value.label)