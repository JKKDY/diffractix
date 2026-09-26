from __future__ import annotations

import ast
import inspect
import textwrap
import math


from collections.abc import Callable
from dataclasses import dataclass, field
import autograd.numpy as np

from diffractix.graph import Comparison, Literal, Node, Relation

from .utils import callable_arity


def describe_callable(func: Callable) -> str:
    name = getattr(func, "__name__", None)

    if name and name != "<lambda>":
        return name

    if name == "<lambda>":
        try:
            lines, start_line = inspect.getsourcelines(func)
            source = textwrap.dedent("".join(lines))
            tree = ast.parse(source)

            target_line = func.__code__.co_firstlineno

            lambdas = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Lambda)
                and start_line + node.lineno - 1 == target_line
            ]

            if not lambdas:
                lambdas = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Lambda)
                ]

            if lambdas:
                return ast.unparse(lambdas[0])

        except (OSError, TypeError, SyntaxError):
            pass

        return "<lambda>"

    return type(func).__name__



@dataclass(frozen=True)
class Constraint:
    evaluate: Node | Callable
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    label: str | None = None
    _repr: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.evaluate, Node) and not callable(self.evaluate):
            raise TypeError("evaluate must be a Node or callable.")

        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound.")

        object.__setattr__(self, "_repr", self._make_repr())

    def _make_repr(self) -> str:
        if self.label is not None:
            return self.label

        if isinstance(self.evaluate, Node):
            expr = repr(self.evaluate)
        else:
            desc = describe_callable(self.evaluate)
            expr = f"({desc})" if desc.startswith("lambda ") else desc

        has_low = not (math.isinf(self.lower_bound) and self.lower_bound < 0)
        has_high = not (math.isinf(self.upper_bound) and self.upper_bound > 0)

        lo, hi = self.lower_bound, self.upper_bound
        match (has_low, has_high):
            case (True, True) if lo == hi:
                return f"{expr} == {lo:g}"
            case (True, True):
                return f"{lo:g} <= {expr} <= {hi:g}"
            case (True, False):
                return f"{expr} >= {lo:g}"
            case (False, True):
                return f"{expr} <= {hi:g}"
            case _:
                return expr

    def __str__(self):
        return self._repr



def _from_comparison(comparison) -> Constraint:
    lhs, rhs, rel = comparison.left, comparison.right, comparison.relation

    if isinstance(rhs, Literal):
        if rel is Relation.LT:
            return Constraint(lhs, upper_bound=math.nextafter(rhs.value, -math.inf))
        if rel is Relation.LE:
            return Constraint(lhs, upper_bound=rhs.value)
        if rel is Relation.GT:
            return Constraint(lhs, lower_bound=math.nextafter(rhs.value, math.inf))
        if rel is Relation.GE:
            return Constraint(lhs, lower_bound=rhs.value)
        if rel is Relation.EQ:
            return Constraint(lhs, rhs.value, rhs.value)

    if isinstance(lhs, Literal):
        if rel is Relation.LT:
            return Constraint(rhs, lower_bound=math.nextafter(lhs.value, math.inf))
        if rel is Relation.LE:
            return Constraint(rhs, lower_bound=lhs.value)
        if rel is Relation.GT:
            return Constraint(rhs, upper_bound=math.nextafter(lhs.value, -math.inf))
        if rel is Relation.GE:
            return Constraint(rhs, upper_bound=lhs.value)
        if rel is Relation.EQ:
            return Constraint(rhs, lhs.value, lhs.value)

    residual = lhs - rhs

    if rel is Relation.LT:
        return Constraint(residual, upper_bound=math.nextafter(0.0, -math.inf))
    if rel is Relation.LE:
        return Constraint(residual, upper_bound=0.0)
    if rel is Relation.GT:
        return Constraint(residual, lower_bound=math.nextafter(0.0, math.inf))
    if rel is Relation.GE:
        return Constraint(residual, lower_bound=0.0)
    if rel is Relation.EQ:
        return Constraint(residual, 0.0, 0.0)

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