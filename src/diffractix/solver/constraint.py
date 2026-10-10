from __future__ import annotations

import ast
import inspect
import textwrap
import math

from fractions import Fraction

from scipy.optimize import linprog

from collections.abc import Callable
from dataclasses import dataclass, field
import autograd.numpy as np

from diffractix.graph import (
    Comparison,
    InputNode,
    Literal,
    Node,
    Parameter,
    Relation,
)

from diffractix.graph.node import BinaryOp, UnaryOp
from diffractix.graph.ops import Op

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




def _direct_parameter(node):
    """Resolve InputNode indirection to a direct Parameter."""
    seen = set()

    while isinstance(node, InputNode):
        if id(node) in seen or node.node is None:
            return None

        seen.add(id(node))
        node = node.node

    return node if isinstance(node, Parameter) else None



def remove_bound_redundant_constraints(constraints, *, parameter_info, parameter_lower, parameter_upper):
    """Remove constraints implied by the canonical parameter bounds."""
    if len(parameter_lower) != len(parameter_upper):
        raise ValueError("Parameter bounds must have matching lengths.")

    reduced = []

    for constraint in constraints:
        parameter = _direct_parameter(constraint.evaluate)

        if parameter is None:
            reduced.append(constraint)
            continue

        info = parameter_info.get(id(parameter))

        if info is None:
            reduced.append(constraint)
            continue

        if info.parameter_index is None:
            domain_lower = domain_upper = info.value
        else:
            index = info.parameter_index

            if not 0 <= index < len(parameter_lower):
                raise ValueError("Invalid parameter index.")

            domain_lower = parameter_lower[index]
            domain_upper = parameter_upper[index]

        lower_redundant = domain_lower >= constraint.lower_bound
        upper_redundant = domain_upper <= constraint.upper_bound

        # Entire constraint is implied by the parameter domain.
        if lower_redundant and upper_redundant:
            continue

        # Preserve equality constraints unless fully redundant.
        if constraint.lower_bound == constraint.upper_bound:
            reduced.append(constraint)
            continue

        lower = -np.inf if lower_redundant else constraint.lower_bound
        upper = np.inf if upper_redundant else constraint.upper_bound

        if lower == constraint.lower_bound and upper == constraint.upper_bound:
            reduced.append(constraint)
            continue

        reduced.append(
            Constraint(
                evaluate=constraint.evaluate,
                lower_bound=lower,
                upper_bound=upper,
                label=constraint.label,
            )
        )

    return tuple(reduced)



def _affine_form(node, parameter_info, n):
    """Return (a, b) for an affine expression, or None."""
    visiting = set()
    memo = {}

    def form(a, b):
        try:
            a = np.asarray(a, dtype=float)
            b = float(b)
        except (TypeError, ValueError, OverflowError):
            return None

        if (
            a.shape != (n,)
            or not np.all(np.isfinite(a))
            or not math.isfinite(b)
        ):
            return None

        return a, b

    def visit(node):
        key = id(node)

        if key in memo:
            return memo[key]

        if key in visiting:
            return None

        visiting.add(key)
        result = None

        try:
            if isinstance(node, InputNode):
                if node.node is not None:
                    result = visit(node.node)

            elif isinstance(node, Literal):
                result = form(np.zeros(n), node.value)

            elif isinstance(node, Parameter):
                info = parameter_info.get(id(node))

                if info is not None:
                    if info.parameter_index is None:
                        result = form(np.zeros(n), info.value)
                    else:
                        index = info.parameter_index

                        if not 0 <= index < n:
                            raise ValueError(
                                "Invalid canonical parameter index."
                            )

                        a = np.zeros(n)
                        a[index] = 1.0
                        result = (a, 0.0)

            elif isinstance(node, UnaryOp):
                operand = visit(node.operand)

                if operand is not None:
                    a, b = operand

                    if node.op is Op.NEG:
                        result = form(-a, -b)
                    elif node.op is Op.POS:
                        result = operand
                    elif not np.any(a):
                        try:
                            result = form(
                                np.zeros(n),
                                node.op.func(b),
                            )
                        except (ArithmeticError, TypeError, ValueError):
                            pass

            elif isinstance(node, BinaryOp):
                left = visit(node.left)
                right = visit(node.right)

                if left is not None and right is not None:
                    a, b = left
                    c, d = right

                    if node.op is Op.ADD:
                        result = form(a + c, b + d)

                    elif node.op is Op.SUB:
                        result = form(a - c, b - d)

                    elif node.op is Op.MUL:
                        if not np.any(a):
                            result = form(b * c, b * d)
                        elif not np.any(c):
                            result = form(d * a, d * b)

                    elif (
                        node.op is Op.DIV
                        and not np.any(c)
                        and d != 0
                    ):
                        result = form(a / d, b / d)

                    elif (
                        node.op is Op.POW
                        and not np.any(c)
                        and d == 1
                    ):
                        result = left

                    elif not np.any(a) and not np.any(c):
                        try:
                            result = form(
                                np.zeros(n),
                                node.op.func(b, d),
                            )
                        except (ArithmeticError, TypeError, ValueError):
                            pass

        finally:
            visiting.remove(key)

        memo[key] = result
        return result

    return visit(node) if isinstance(node, Node) else None


def remove_affine_redundant_constraints(constraints, *, parameter_info, parameter_lower, parameter_upper):
    """Remove affine constraints implied by other affine constraints and bounds."""
    lower = np.asarray(parameter_lower, dtype=float)
    upper = np.asarray(parameter_upper, dtype=float)

    if (
        lower.ndim != 1
        or lower.shape != upper.shape
        or np.any(lower > upper)
    ):
        raise ValueError("Invalid canonical parameter bounds.")

    retained = list(constraints)
    forms = [
        _affine_form(c.evaluate, parameter_info, len(lower))
        for c in retained
    ]

    bounds = list(zip(lower, upper))

    def implied(index, side):
        constraint = retained[index]
        a, b = forms[index]

        bound = (
            constraint.lower_bound
            if side == "lower"
            else constraint.upper_bound
        )

        if not np.isfinite(bound):
            return bound == (
                -np.inf if side == "lower" else np.inf
            )

        # Express the candidate as row @ theta <= rhs.
        row = -a if side == "lower" else a
        rhs = b - bound if side == "lower" else bound - b

        if not np.isfinite(rhs):
            return False

        # Constant constraint.
        if not np.any(row):
            return rhs >= 0

        rows = []
        limits = []

        for j, (other, affine) in enumerate(
            zip(retained, forms)
        ):
            if other is None or affine is None:
                continue

            other_a, other_b = affine
            equality = other.lower_bound == other.upper_bound

            # Exclude the candidate side itself.
            # For equalities, exclude both sides.
            if j == index and equality:
                continue

            if (
                other.lower_bound > -np.inf
                and (j != index or side == "upper")
            ):
                rows.append(-other_a)
                limits.append(
                    other_b - other.lower_bound
                )

            if (
                other.upper_bound < np.inf
                and (j != index or side == "lower")
            ):
                rows.append(other_a)
                limits.append(
                    other.upper_bound - other_b
                )

        # Exact matching inequalities: avoid LP tolerance issues.
        for other_row, other_rhs in zip(rows, limits):
            if (
                np.array_equal(other_row, row)
                and other_rhs <= rhs
            ):
                return True

        # Maximize row @ theta over the remaining feasible region.
        result = linprog(
            c=-row,
            A_ub=np.asarray(rows) if rows else None,
            b_ub=np.asarray(limits) if limits else None,
            bounds=bounds,
            method="highs",
        )

        # Unbounded, infeasible, or unsuccessful: preserve constraint.
        if not result.success or not np.isfinite(result.fun):
            return False

        maximum = -result.fun

        # LP solutions are approximate. Require clear numerical slack.
        safety = 1e-7 * max(
            1.0,
            abs(rhs),
            abs(maximum),
        )

        return maximum <= rhs - safety

    for i, constraint in enumerate(retained):
        if forms[i] is None:
            continue

        # Keep equality constraints intact unless fully redundant.
        if constraint.lower_bound == constraint.upper_bound:
            if implied(i, "lower") and implied(i, "upper"):
                retained[i] = None
            continue

        lower_redundant = implied(i, "lower")

        if lower_redundant:
            retained[i] = Constraint(
                evaluate=constraint.evaluate,
                lower_bound=-np.inf,
                upper_bound=constraint.upper_bound,
                label=constraint.label,
            )

        # Test the upper side against the now-retained lower side.
        upper_redundant = implied(i, "upper")

        if upper_redundant:
            retained[i] = Constraint(
                evaluate=constraint.evaluate,
                lower_bound=retained[i].lower_bound,
                upper_bound=np.inf,
                label=constraint.label,
            )

        if (
            retained[i].lower_bound == -np.inf
            and retained[i].upper_bound == np.inf
        ):
            retained[i] = None

    return tuple(
        constraint
        for constraint in retained
        if constraint is not None
    )


def remove_duplicate_constraints(constraints):
    retained = []

    for constraint in constraints:
        if (
            constraint.lower_bound == -np.inf
            and constraint.upper_bound == np.inf
        ):
            continue

        if any(
            other.evaluate is constraint.evaluate
            and other.lower_bound >= constraint.lower_bound
            and other.upper_bound <= constraint.upper_bound
            for other in retained
        ):
            continue

        retained = [
            other
            for other in retained
            if not (
                other.evaluate is constraint.evaluate
                and constraint.lower_bound >= other.lower_bound
                and constraint.upper_bound <= other.upper_bound
            )
        ]

        retained.append(constraint)

    return tuple(retained)