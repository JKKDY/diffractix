from __future__ import annotations

import ast
import inspect
import textwrap
import math

from fractions import Fraction

from scipy.optimize import linprog

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import numpy as np

from diffractix.graph import (
    Comparison,
    Literal,
    Node,
    Parameter,
    Program,
    Relation,
    Symbol,
    SymbolicControlFlowError,
    precompile,
    walk_ast,
)

from diffractix.graph.ops import Op
from diffractix.graph.program import Input, Register, Constant, UnaryInstruction

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


@dataclass(frozen=True)
class ConstraintDescriptor:
    position: int
    lower_bound: float
    upper_bound: float
    label: str | None
    description: str



def _from_comparison(comparison) -> Constraint:
    lhs, rhs, rel = comparison.left, comparison.right, comparison.relation

    if isinstance(rhs, Literal) and np.ndim(rhs.value) == 0:
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

    if isinstance(lhs, Literal) and np.ndim(lhs.value) == 0:
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


def prepare_constraints(constraints, *, tracer, bindings):
    """Trace and precompile constraints, retaining dynamic callables."""
    from diffractix.simulation.result import ResultSymbolKey

    roots = []
    descriptors = []
    callables = []
    for position, requirement in enumerate(constraints):
        constraint = normalize_to_constraint(requirement)
        expression = constraint.evaluate
        if not isinstance(expression, Node):
            if tracer is None:
                callables.append((position, constraint))
                continue
            try:
                expression = expression(tracer)
            except SymbolicControlFlowError:
                callables.append((position, constraint))
                continue
            if not isinstance(expression, Node):
                expression = Node._make_literal(expression)

        has_inputs = False
        for node in walk_ast((expression,), bindings):
            if isinstance(node, Symbol) and node.key not in bindings:
                if not isinstance(node.key, ResultSymbolKey):
                    raise ValueError(f"Constraint {position} has an unresolved Symbol binding: {node.key!r}.")
                has_inputs = True
            if isinstance(node, Parameter) and node.is_variable:
                has_inputs = True

        if not has_inputs:
            value = np.asarray(expression.evaluate(bindings))
            description = constraint.label or f"Constraint {position}"
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{description} has a non-finite constant value.")
            if not np.all((value >= constraint.lower_bound) & (value <= constraint.upper_bound)):
                raise ValueError(f"{description} is an unsatisfiable constant constraint: value {value} lies outside [{constraint.lower_bound}, {constraint.upper_bound}].")
            continue

        roots.append(expression)
        descriptors.append(ConstraintDescriptor(position, constraint.lower_bound, constraint.upper_bound, constraint.label, str(constraint)))

    return precompile(roots, bindings), tuple(descriptors), tuple(callables)


def reduce_constraints(
    program: Program,
    descriptors: Sequence[ConstraintDescriptor],
    *,
    parameter_lower: Sequence[float],
    parameter_upper: Sequence[float],
) -> tuple[tuple[int, ...], tuple[ConstraintDescriptor, ...]]:
    """Select program outputs and effective bounds for the current solve.

    Each returned descriptor describes the corresponding selected output.
    Descriptor positions retain their original declaration positions.
    The full program and original descriptors are preserved for reuse.
    """
    if len(descriptors) != len(program._outputs):
        raise ValueError("Expected one descriptor per constraint program output.")
    parameter_lower = np.asarray(parameter_lower, dtype=float)
    parameter_upper = np.asarray(parameter_upper, dtype=float)
    expected_shape = (len(program.parameter_indices),)
    if parameter_lower.shape != expected_shape or parameter_upper.shape != expected_shape:
        raise ValueError(f"Parameter bounds must have shape {expected_shape}.")
    if (
        np.any(np.isnan(parameter_lower)) or np.any(np.isnan(parameter_upper))
        or np.any(parameter_lower > parameter_upper)
        or np.any(parameter_lower == np.inf) or np.any(parameter_upper == -np.inf)
    ):
        raise ValueError("Invalid parameter bounds.")
    for descriptor in descriptors:
        if (
            math.isnan(descriptor.lower_bound) or math.isnan(descriptor.upper_bound)
            or descriptor.lower_bound > descriptor.upper_bound
        ):
            raise ValueError("Invalid constraint bounds.")

    output_indices = tuple(range(len(descriptors)))
    active_descriptors = tuple(descriptors)

    output_indices, active_descriptors = _reduce_duplicate_outputs(
        program, output_indices, active_descriptors,
    )
    if not output_indices:
        return output_indices, active_descriptors

    forms = _program_affine_forms(program)
    output_indices, active_descriptors = _reduce_box_constraints(
        forms, output_indices, active_descriptors,
        parameter_lower, parameter_upper,
    )
    if not output_indices:
        return output_indices, active_descriptors

    output_indices, active_descriptors = _reduce_interval_constraints(
        program, output_indices, active_descriptors,
        parameter_lower, parameter_upper,
    )
    if not output_indices:
        return output_indices, active_descriptors

    output_indices, active_descriptors = _reduce_equalities(
        forms, output_indices, active_descriptors,
    )
    output_indices, active_descriptors = _reduce_affine_constraints(
        forms, output_indices, active_descriptors,
        parameter_lower, parameter_upper,
    )
    return output_indices, active_descriptors


def _program_output_shapes(program):
    """Infer output shapes where all inputs have known shapes."""
    registers = {}

    def read(operand):
        if isinstance(operand, Register):
            return registers[operand.index]
        if isinstance(operand, Input):
            return () if operand.index < len(program.parameter_indices) else None
        return np.shape(operand.value)

    for instruction in program._instructions:
        if isinstance(instruction, UnaryInstruction):
            shape = read(instruction.src)
        else:
            left, right = read(instruction.left), read(instruction.right)
            shape = None
            if left is not None and right is not None:
                try:
                    shape = np.broadcast_shapes(left, right)
                except ValueError:
                    pass
        registers[instruction.dst] = shape
    return tuple(read(output) for output in program._outputs)


def _reduce_duplicate_outputs(program, output_indices, descriptors):
    """Combine bounds on structurally identical computations."""
    values = {}
    registers = {}

    def identify(operand):
        if isinstance(operand, Register):
            return registers[operand.index]
        if isinstance(operand, Input):
            key = (Input, operand.index)
        else:
            value = operand.value
            if type(value) in (bool, int) or (type(value) in (float, complex) and np.isfinite(value)):
                key = (Constant, type(value), repr(value))
            elif isinstance(value, np.generic) and value.dtype.kind in "biufc" and np.isfinite(value):
                key = (Constant, value.dtype.str, value.tobytes())
            elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc" and np.all(np.isfinite(value)):
                key = (Constant, value.dtype.str, value.shape, value.tobytes())
            else:
                key = object()
        return number(key)

    def number(key):
        if key not in values:
            values[key] = len(values)
        return values[key]

    for instruction in program._instructions:
        if isinstance(instruction, UnaryInstruction):
            key = (instruction.op, identify(instruction.src))
        else:
            key = (instruction.op, identify(instruction.left), identify(instruction.right))
        registers[instruction.dst] = number(key)
    output_keys = tuple(identify(output) for output in program._outputs)
    output_shapes = _program_output_shapes(program)

    retained_indices = []
    retained = []
    representatives = {}
    for output_index, descriptor in zip(output_indices, descriptors):
        shape = output_shapes[output_index]
        if shape is not None and 0 in shape:
            continue
        if descriptor.lower_bound == -np.inf and descriptor.upper_bound == np.inf:
            continue
        key = output_keys[output_index]
        if key not in representatives:
            representatives[key] = len(retained)
            retained_indices.append(output_index)
            retained.append(descriptor)
            continue
        index = representatives[key]
        previous = retained[index]
        lower = max(previous.lower_bound, descriptor.lower_bound)
        upper = min(previous.upper_bound, descriptor.upper_bound)
        if lower > upper:
            # An unknown runtime array could be empty, satisfying both bounds.
            if shape is None:
                retained_indices.append(output_index)
                retained.append(descriptor)
                continue
            raise ValueError(f"Conflicting bounds on duplicate constraints {previous.position} and {descriptor.position}.")
        if lower != previous.lower_bound or upper != previous.upper_bound:
            retained[index] = replace(previous, lower_bound=lower, upper_bound=upper)
    return tuple(retained_indices), tuple(retained)


def _program_affine_forms(program):
    """Interpret program outputs as rational affine forms, or None if unknown."""
    size = len(program.parameter_indices)
    zero = (Fraction(0),) * size
    registers = {}

    def read(operand):
        if isinstance(operand, Register):
            return registers[operand.index]
        if isinstance(operand, Input):
            if operand.index >= size:
                return None
            coefficients = list(zero)
            coefficients[operand.index] = Fraction(1)
            return tuple(coefficients), Fraction(0)
        value = _rational(operand.value)
        return None if value is None else (zero, value)

    for instruction in program._instructions:
        result = None
        if isinstance(instruction, UnaryInstruction):
            operand = read(instruction.src)
            if operand is not None:
                coefficients, offset = operand
                if instruction.op is Op.POS:
                    result = operand
                elif instruction.op is Op.NEG:
                    result = tuple(-value for value in coefficients), -offset
                elif instruction.op is Op.ABS and coefficients == zero:
                    result = zero, abs(offset)
        else:
            left = read(instruction.left)
            right = read(instruction.right)
            if left is not None and right is not None:
                a, b = left
                c, d = right
                if instruction.op is Op.ADD:
                    result = tuple(x + y for x, y in zip(a, c)), b + d
                elif instruction.op is Op.SUB:
                    result = tuple(x - y for x, y in zip(a, c)), b - d
                elif instruction.op is Op.MUL:
                    if a == zero:
                        result = tuple(b * value for value in c), b * d
                    elif c == zero:
                        result = tuple(d * value for value in a), d * b
                elif instruction.op is Op.DIV and c == zero and d != 0:
                    result = tuple(value / d for value in a), b / d
                elif instruction.op is Op.POW and c == zero:
                    if d == 0:
                        result = zero, Fraction(1)
                    elif d == 1:
                        result = left
                elif instruction.op is Op.MIN and a == c == zero:
                    result = zero, min(b, d)
                elif instruction.op is Op.MAX and a == c == zero:
                    result = zero, max(b, d)
        registers[instruction.dst] = result
    return tuple(read(output) for output in program._outputs)


def _remove_implied_bounds(descriptor, lower_implied, upper_implied):
    """Remove proven bound sides, keeping equalities whole unless fully implied."""
    if lower_implied and upper_implied:
        return None
    if descriptor.lower_bound == descriptor.upper_bound:
        return descriptor
    lower = -np.inf if lower_implied else descriptor.lower_bound
    upper = np.inf if upper_implied else descriptor.upper_bound
    if lower == descriptor.lower_bound and upper == descriptor.upper_bound:
        return descriptor
    return replace(descriptor, lower_bound=lower, upper_bound=upper)


def _reduce_box_constraints(forms, output_indices, descriptors, parameter_lower, parameter_upper):
    """Remove affine constraint sides implied by parameter bounds."""
    retained_indices = []
    retained = []
    for output_index, descriptor in zip(output_indices, descriptors):
        form = forms[output_index]
        if form is None:
            retained_indices.append(output_index)
            retained.append(descriptor)
            continue
        coefficients, offset = form
        negated = tuple(-value for value in coefficients)
        lower = _rational(descriptor.lower_bound)
        upper = _rational(descriptor.upper_bound)
        lower_implied = descriptor.lower_bound == -np.inf or (
            lower is not None and _box_implies(negated, offset - lower, parameter_lower, parameter_upper)
        )
        upper_implied = descriptor.upper_bound == np.inf or (
            upper is not None and _box_implies(coefficients, upper - offset, parameter_lower, parameter_upper)
        )
        effective = _remove_implied_bounds(descriptor, lower_implied, upper_implied)
        if effective is None:
            continue
        if not any(coefficients):
            raise ValueError(f"Constraint {descriptor.position} is an unsatisfiable constant constraint.")
        retained_indices.append(output_index)
        retained.append(effective)
    return tuple(retained_indices), tuple(retained)


def _interval_product(left, right):
    products = [first * second for first in left for second in right]
    return min(products), max(products)


def _program_intervals(program, parameter_lower, parameter_upper):
    """Bound supported real operations using finite rational intervals."""
    registers = {}

    def read(operand):
        if isinstance(operand, Register):
            return registers[operand.index]
        if isinstance(operand, Input):
            if operand.index >= len(program.parameter_indices):
                return None
            lower = _rational(parameter_lower[operand.index])
            upper = _rational(parameter_upper[operand.index])
            return None if lower is None or upper is None else (lower, upper)
        value = _rational(operand.value)
        return None if value is None else (value, value)

    for instruction in program._instructions:
        result = None
        if isinstance(instruction, UnaryInstruction):
            operand = read(instruction.src)
            if operand is not None:
                lower, upper = operand
                if instruction.op is Op.POS:
                    result = operand
                elif instruction.op is Op.NEG:
                    result = -upper, -lower
                elif instruction.op is Op.ABS:
                    minimum = Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
                    result = minimum, max(abs(lower), abs(upper))
        else:
            left = read(instruction.left)
            right = read(instruction.right)
            if left is not None and right is not None:
                lower, upper = left
                other_lower, other_upper = right
                if instruction.op is Op.ADD:
                    result = lower + other_lower, upper + other_upper
                elif instruction.op is Op.SUB:
                    result = lower - other_upper, upper - other_lower
                elif instruction.op is Op.MUL:
                    result = _interval_product(left, right)
                elif instruction.op is Op.DIV and not other_lower <= 0 <= other_upper:
                    result = _interval_product(left, (1 / other_upper, 1 / other_lower))
                elif instruction.op is Op.POW and other_lower == other_upper:
                    if other_lower == 0:
                        result = Fraction(1), Fraction(1)
                    elif other_lower == 1:
                        result = left
                    elif other_lower == 2:
                        minimum = Fraction(0) if lower <= 0 <= upper else min(lower ** 2, upper ** 2)
                        result = minimum, max(lower ** 2, upper ** 2)
                elif instruction.op is Op.MIN:
                    result = min(lower, other_lower), min(upper, other_upper)
                elif instruction.op is Op.MAX:
                    result = max(lower, other_lower), max(upper, other_upper)
        registers[instruction.dst] = result
    return tuple(read(output) for output in program._outputs)


def _reduce_interval_constraints(program, output_indices, descriptors, parameter_lower, parameter_upper):
    """Remove constraint sides implied by conservative numerical ranges."""
    intervals = _program_intervals(program, parameter_lower, parameter_upper)
    retained_indices = []
    retained = []
    for output_index, descriptor in zip(output_indices, descriptors):
        interval = intervals[output_index]
        if interval is None:
            retained_indices.append(output_index)
            retained.append(descriptor)
            continue
        minimum, maximum = interval
        lower = _rational(descriptor.lower_bound)
        upper = _rational(descriptor.upper_bound)
        if (lower is not None and maximum < lower) or (upper is not None and minimum > upper):
            raise ValueError(f"Constraint {descriptor.position} is infeasible within the parameter bounds.")
        lower_implied = descriptor.lower_bound == -np.inf or (lower is not None and minimum >= lower)
        upper_implied = descriptor.upper_bound == np.inf or (upper is not None and maximum <= upper)
        effective = _remove_implied_bounds(descriptor, lower_implied, upper_implied)
        if effective is None:
            continue
        retained_indices.append(output_index)
        retained.append(effective)
    return tuple(retained_indices), tuple(retained)


def _reduce_equalities(forms, output_indices, descriptors):
    """Keep the first independent affine equalities using exact row elimination."""
    basis = {}
    retained_indices = []
    retained = []
    for output_index, descriptor in zip(output_indices, descriptors):
        form = forms[output_index]
        bound = _rational(descriptor.lower_bound)
        if form is not None and bound is not None and descriptor.lower_bound == descriptor.upper_bound:
            coefficients, offset = form
            row = [*coefficients, bound - offset]
            for pivot in sorted(basis):
                multiplier = row[pivot]
                if multiplier != 0:
                    row = [value - multiplier * other for value, other in zip(row, basis[pivot])]
            pivot = next((index for index, value in enumerate(row[:-1]) if value != 0), None)
            if pivot is None:
                if row[-1] != 0:
                    raise ValueError(f"Constraint {descriptor.position} conflicts with the preceding affine equalities.")
                continue
            divisor = row[pivot]
            basis[pivot] = tuple(value / divisor for value in row)
        retained_indices.append(output_index)
        retained.append(descriptor)
    return tuple(retained_indices), tuple(retained)


def _reduce_affine_constraints(output_forms, output_indices, descriptors, parameter_lower, parameter_upper):
    """Remove affine constraint sides implied by other retained constraints."""
    forms = [output_forms[index] for index in output_indices]
    if all(form is None for form in forms):
        return output_indices, descriptors
    retained = list(descriptors)
    rows = _affine_inequalities(
        retained, forms, exclude_index=-1, exclude_side=None,
        lower=parameter_lower, upper=parameter_upper,
    )
    _check_affine_feasibility(rows, len(parameter_lower))

    def implied(index, side):
        descriptor = retained[index]
        coefficients, offset = forms[index]
        if side == "lower":
            if descriptor.lower_bound == -np.inf:
                return True
            bound = _rational(descriptor.lower_bound)
            if bound is None:
                return False
            target = tuple(-value for value in coefficients)
            rhs = offset - bound
        else:
            if descriptor.upper_bound == np.inf:
                return True
            bound = _rational(descriptor.upper_bound)
            if bound is None:
                return False
            target = coefficients
            rhs = bound - offset
        available = _affine_inequalities(
            retained, forms, exclude_index=index, exclude_side=side,
            lower=parameter_lower, upper=parameter_upper,
        )
        return _affine_certificate(available, target, rhs)

    for index, descriptor in enumerate(retained):
        if forms[index] is None:
            continue
        if descriptor.lower_bound == descriptor.upper_bound:
            if implied(index, "lower") and implied(index, "upper"):
                retained[index] = None
            continue
        lower = -np.inf if implied(index, "lower") else descriptor.lower_bound
        if lower != descriptor.lower_bound:
            retained[index] = replace(descriptor, lower_bound=lower)
        upper = np.inf if implied(index, "upper") else descriptor.upper_bound
        if lower == -np.inf and upper == np.inf:
            retained[index] = None
        elif lower != descriptor.lower_bound or upper != descriptor.upper_bound:
            retained[index] = replace(descriptor, lower_bound=lower, upper_bound=upper)

    pairs = [(index, descriptor) for index, descriptor in zip(output_indices, retained) if descriptor is not None]
    return tuple(index for index, _ in pairs), tuple(descriptor for _, descriptor in pairs)


def _check_affine_feasibility(rows, size):
    """Verify a rational infeasibility certificate when HiGHS reports a conflict."""
    if not rows or size == 0:
        return
    try:
        result = linprog(
            c=np.zeros(size),
            A_ub=np.asarray([[float(value) for value in coefficients] for coefficients, _ in rows]),
            b_ub=np.asarray([float(rhs) for _, rhs in rows]),
            bounds=(None, None),
            method="highs",
        )
    except (ValueError, OverflowError, FloatingPointError):
        return
    if result.status != 2:
        return

    # A nonnegative combination yielding 0 <= -1 proves infeasibility.
    augmented_rows = [(tuple(coefficients) + (rhs,), Fraction(0)) for coefficients, rhs in rows]
    target = (Fraction(0),) * size + (Fraction(-1),)
    try:
        certificate = linprog(
            c=np.zeros(len(rows)),
            A_eq=np.asarray([[float(row[j]) for row, _ in augmented_rows] for j in range(size + 1)]),
            b_eq=np.asarray(target, dtype=float),
            bounds=(0, None),
            method="highs",
        )
    except (ValueError, OverflowError, FloatingPointError):
        return
    if not certificate.success or certificate.x is None or not np.all(np.isfinite(certificate.x)):
        return
    weights = _recover_exact_weights(augmented_rows, target, certificate.x)
    if weights is None or any(weight < 0 for weight in weights):
        return
    if all(
        sum(weight * row[j] for weight, (row, _) in zip(weights, augmented_rows)) == target[j]
        for j in range(size + 1)
    ):
        raise ValueError("Affine constraints and parameter bounds are infeasible.")


def _rational(value):
    """Exact rational representation of a finite numeric scalar, or None."""
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    try:
        return Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


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
    weights = [Fraction(0)] * len(active)
    for r, col in enumerate(pivot_columns):
        weights[col] = matrix[r][-1]

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
