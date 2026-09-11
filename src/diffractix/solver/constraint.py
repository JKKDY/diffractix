from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as np


@dataclass(frozen=True)
class Constraint:
    """Hard nonlinear constraint evaluated between lower and upper bounds."""

    evaluate: object
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    label: str | None = None

    def __post_init__(self):
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound.")



def normalize_to_constraint(comparison):
    lhs, rhs, rel = comparison.left, comparison.right, comparison.relation

    if isinstance(rhs, Literal):
        if rel is Relation.LE: return Constraint(lhs, upper_bound=rhs.value)
        if rel is Relation.GE: return Constraint(lhs, lower_bound=rhs.value)
        if rel is Relation.EQ: return Constraint(lhs, lower_bound=rhs.value, upper_bound=rhs.value)

    if isinstance(lhs, Literal):
        if rel is Relation.LE: return Constraint(rhs, lower_bound=lhs.value)
        if rel is Relation.GE: return Constraint(rhs, upper_bound=lhs.value)
        if rel is Relation.EQ: return Constraint(rhs, lower_bound=lhs.value, upper_bound=lhs.value)

    residual = lhs - rhs
    if rel is Relation.LE: return Constraint(residual, upper_bound=0.0)
    if rel is Relation.GE: return Constraint(residual, lower_bound=0.0)
    if rel is Relation.EQ: return Constraint(residual, lower_bound=0.0, upper_bound=0.0)