from __future__ import annotations

import ast
import inspect
import textwrap
import math

from fractions import Fraction
from numbers import Real

from scipy.optimize import linprog

from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np

from diffractix.graph import (
    Comparison,
    InputNode,
    Literal,
    Node,
    Parameter,
    Relation,
    Symbol,
)

from diffractix.graph.node import BinaryOp, UnaryOp
from diffractix.graph.ops import Op
from diffractix.graph.utils import ast_equal

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

        for name, bound in (("lower_bound", self.lower_bound), ("upper_bound", self.upper_bound)):
            if math.isnan(bound):
                raise ValueError(f"{name} must not be NaN.")

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



def _rational(value):
    """Exact rational representation of a finite numeric scalar, or None."""
    try:
        return Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _affine_form(node, parameter_info, n, compile_context=None):
    """Return (coefficients, offset) as Fractions, or None if not provably affine."""
    zero = (Fraction(0),) * n
    visiting, memo = set(), {}

    def visit(current):
        key = id(current)
        if key in memo:
            return memo[key]
        if key in visiting:
            return None

        visiting.add(key)
        result = None
        try:
            if isinstance(current, InputNode):
                if current.node is not None:
                    result = visit(current.node)

            elif isinstance(current, Literal):
                if (value := _rational(current.value)) is not None:
                    result = (zero, value)

            elif isinstance(current, Symbol):
                if compile_context is not None and current.key in compile_context:
                    binding = compile_context[current.key]
                    if isinstance(binding, Node):
                        result = visit(binding)
                    elif isinstance(binding, Real) and not isinstance(binding, bool):
                        if (value := _rational(binding)) is not None:
                            result = (zero, value)

            elif isinstance(current, Parameter):
                info = parameter_info.get(id(current))
                if info is not None:
                    if info.parameter_index is None and (value := _rational(info.value)) is not None:
                        result = (zero, value)
                    elif info.parameter_index is not None:
                        index = info.parameter_index
                        if not 0 <= index < n:
                            raise ValueError('Invalid canonical parameter index.')
                        coefficients = list(zero)
                        coefficients[index] = Fraction(1)
                        result = (tuple(coefficients), Fraction(0))

            elif isinstance(current, UnaryOp):
                if (operand := visit(current.operand)) is not None:
                    a, b = operand
                    if current.op is Op.POS:
                        result = operand
                    elif current.op is Op.NEG:
                        result = (tuple(-v for v in a), -b)
                    elif current.op is Op.ABS and a == zero:
                        result = (zero, abs(b))

            elif isinstance(current, BinaryOp):
                left, right = visit(current.left), visit(current.right)
                if left is not None and right is not None:
                    a, b = left
                    c, d = right
                    if current.op is Op.ADD:
                        result = (tuple(x + y for x, y in zip(a, c)), b + d)
                    elif current.op is Op.SUB:
                        result = (tuple(x - y for x, y in zip(a, c)), b - d)
                    elif current.op is Op.MUL:
                        if a == zero:
                            result = (tuple(b * x for x in c), b * d)
                        elif c == zero:
                            result = (tuple(d * x for x in a), d * b)
                    elif current.op is Op.DIV and c == zero and d != 0:
                        result = (tuple(x / d for x in a), b / d)
                    elif current.op is Op.POW and c == zero:
                        if d == 0:
                            result = (zero, Fraction(1))
                        elif d == 1:
                            result = left
                    elif current.op is Op.MAX and a == c == zero:
                        result = (zero, max(b, d))
                    elif current.op is Op.MIN and a == c == zero:
                        result = (zero, min(b, d))
        finally:
            visiting.remove(key)
        memo[key] = result
        return result

    return visit(node) if isinstance(node, Node) else None


def _affine_inequalities(constraints, forms, exclude_index, exclude_side, lower, upper):
    """Build exact inequalities A @ x <= b from retained constraints and box."""
    rows = []
    for i, (constraint, form) in enumerate(zip(constraints, forms)):
        if constraint is None or form is None:
            continue
        a, offset = form
        if i == exclude_index and constraint.lower_bound == constraint.upper_bound:
            continue
        if i != exclude_index or exclude_side != 'upper':
            if (bound := _rational(constraint.upper_bound)) is not None:
                rows.append((a, bound - offset))
        if i != exclude_index or exclude_side != 'lower':
            if (bound := _rational(constraint.lower_bound)) is not None:
                rows.append((tuple(-v for v in a), offset - bound))

    for j, (lo, hi) in enumerate(zip(lower, upper)):
        if (hi := _rational(hi)) is not None:
            row = [Fraction(0)] * len(lower)
            row[j] = Fraction(1)
            rows.append((tuple(row), hi))
        if (lo := _rational(lo)) is not None:
            row = [Fraction(0)] * len(lower)
            row[j] = Fraction(-1)
            rows.append((tuple(row), -lo))
    return rows


def _box_implies(row, rhs, lower, upper):
    """Prove row @ x <= rhs from parameter bounds alone."""
    maximum = Fraction(0)
    for coefficient, lo, hi in zip(row, lower, upper):
        if coefficient == 0:
            continue
        endpoint = hi if coefficient > 0 else lo
        if (value := _rational(endpoint)) is None:
            return False
        maximum += coefficient * value
    return maximum <= rhs


def _recover_exact_weights(rows, target, approximate):
    """Solve A.T @ weights = target exactly on the LP's active support."""
    if not (active := [i for i, value in enumerate(approximate) if value > 0]):
        return None
    matrix = [
        [rows[i][0][j] for i in active] + [target[j]]
        for j in range(len(target))
    ]
    pivot_columns, pivot_row = [], 0
    for col in range(len(active)):
        row = next((r for r in range(pivot_row, len(matrix)) if matrix[r][col]), None)
        if row is None:
            continue
        matrix[pivot_row], matrix[row] = matrix[row], matrix[pivot_row]
        pivot = matrix[pivot_row][col]
        matrix[pivot_row] = [v / pivot for v in matrix[pivot_row]]
        for r in range(len(matrix)):
            if r != pivot_row and (multiplier := matrix[r][col]):
                matrix[r] = [x - multiplier * y for x, y in zip(matrix[r], matrix[pivot_row])]
        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    if any(all(v == 0 for v in row[:-1]) and row[-1] != 0 for row in matrix):
        return None

    # Take free variables as zero; solve pivot multipliers exactly.
    free_columns = [j for j in range(len(active)) if j not in pivot_columns]
    weights = [Fraction(0)] * len(active)
    for r, col in enumerate(pivot_columns):
        weights[col] = matrix[r][-1] - sum(matrix[r][j] * weights[j] for j in free_columns)

    full = [Fraction(0)] * len(rows)
    for i, value in zip(active, weights):
        full[i] = value
    return full


def _affine_certificate(rows, target, rhs):
    """Use HiGHS to propose a Farkas certificate; verify it over rational affine forms."""
    if not rows:
        return False
    if any(a == target and b <= rhs for a, b in rows):
        return True
    if not target:
        return False

    try:
        result = linprog(
            c=np.asarray([float(b) for _, b in rows]),
            A_eq=np.asarray([[float(a[j]) for a, _ in rows] for j in range(len(target))]),
            b_eq=np.asarray([float(v) for v in target]),
            bounds=(0, None),
            method='highs',
        )
    except (ValueError, OverflowError, FloatingPointError):
        return False

    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        return False

    weights = _recover_exact_weights(rows, target, result.x)
    if weights is None or any(w < 0 for w in weights):
        return False

    # Exact Farkas conditions: A.T @ w == target, b @ w <= rhs.
    return (
        all(sum(w * a[j] for w, (a, _) in zip(weights, rows)) == target[j] for j in range(len(target)))
        and sum(w * b for w, (_, b) in zip(weights, rows)) <= rhs
    )


def remove_affine_redundant_constraints(
    constraints, *, parameter_info, parameter_lower, parameter_upper, compile_context=None,
):
    """Remove affine bounds implied under real-arithmetic interpretation of the AST."""
    lower, upper = tuple(parameter_lower), tuple(parameter_upper)
    if len(lower) != len(upper) or any(np.isnan(lo) or np.isnan(hi) or lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError('Invalid canonical parameter bounds.')

    retained = list(constraints)
    forms = [_affine_form(c.evaluate, parameter_info, len(lower), compile_context) for c in retained]

    def implied(i, side):
        constraint = retained[i]
        a, offset = forms[i]
        if side == 'lower':
            if (bound := _rational(constraint.lower_bound)) is None:
                return constraint.lower_bound == -np.inf
            row, rhs = tuple(-v for v in a), offset - bound
        else:
            if (bound := _rational(constraint.upper_bound)) is None:
                return constraint.upper_bound == np.inf
            row, rhs = a, bound - offset

        if _box_implies(row, rhs, lower, upper):
            return True
        rows = _affine_inequalities(retained, forms, i, side, lower, upper)
        return _affine_certificate(rows, row, rhs)

    for i, constraint in enumerate(retained):
        if forms[i] is None:
            continue

        if constraint.lower_bound == constraint.upper_bound:
            if implied(i, 'lower') and implied(i, 'upper'):
                retained[i] = None
            continue

        if implied(i, 'lower') and constraint.lower_bound != -np.inf:
            retained[i] = Constraint(
                constraint.evaluate, 
                lower_bound=-np.inf, 
                upper_bound=constraint.upper_bound, 
                label=constraint.label
            )

        if implied(i, 'upper') and retained[i].upper_bound != np.inf:
            retained[i] = Constraint(
                constraint.evaluate, 
                lower_bound=retained[i].lower_bound, 
                upper_bound=np.inf, 
                label=constraint.label
            )

        if retained[i].lower_bound == -np.inf and retained[i].upper_bound == np.inf:
            retained[i] = None

    return tuple(c for c in retained if c is not None)

    
def _same_evaluator(a, b):
    if a is b:
        return True

    if isinstance(a, Node) and isinstance(b, Node):
        return ast_equal(a, b)

    return False


def remove_duplicate_constraints(constraints):
    retained = []

    for constraint in constraints:
        if (
            constraint.lower_bound == -np.inf
            and constraint.upper_bound == np.inf
        ):
            continue

        if any(
            _same_evaluator(other.evaluate, constraint.evaluate)
            and other.lower_bound >= constraint.lower_bound
            and other.upper_bound <= constraint.upper_bound
            for other in retained
        ):
            continue

        retained = [
            other
            for other in retained
            if not (
                _same_evaluator(other.evaluate, constraint.evaluate)
                and constraint.lower_bound >= other.lower_bound
                and constraint.upper_bound <= other.upper_bound
            )
        ]

        retained.append(constraint)

    return tuple(retained)
