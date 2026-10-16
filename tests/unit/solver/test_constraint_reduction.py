"""Black-box reduction contracts derived from interval and affine mathematics.

The reducer bodies are deliberately not inspected. Feasibility is checked by
numerically evaluating the public AST, independently of the reduction routines.
Assertions allow either representative of equivalent constraints and allow
interval intersections to be merged or retained as separate constraints.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from diffractix.graph import InputNode, Literal, Node, Parameter, Symbol, evaluate_ast
from diffractix.solver.constraint import (
    Constraint,
    normalize_to_constraint,
    remove_affine_redundant_constraints,
    remove_bound_redundant_constraints,
    remove_duplicate_constraints,
)
from diffractix.system.info import ParameterInfo


@dataclass
class Domain:
    parameters: tuple
    parameter_info: dict
    lower: np.ndarray
    upper: np.ndarray

    def reduce(self, function, constraints):
        if function is remove_duplicate_constraints:
            return tuple(function(constraints))
        return tuple(function(
            constraints,
            parameter_info=self.parameter_info,
            parameter_lower=self.lower,
            parameter_upper=self.upper,
        ))


def domain(*bounds):
    parameters, info = [], {}
    for index, (lower, upper) in enumerate(bounds):
        value = min(max(0.0, lower), upper)
        parameter = Parameter(
            value, name=f"x{index}", variable=True,
            lower_bound=lower, upper_bound=upper,
        )
        parameters.append(parameter)
        info[id(parameter)] = metadata(parameter, index)
    return Domain(
        tuple(parameters), info,
        np.array([b[0] for b in bounds]),
        np.array([b[1] for b in bounds]),
    )


def metadata(parameter, index, *, value=None):
    return ParameterInfo(
        parameter_id=id(parameter), name=parameter.name,
        value=parameter.value if value is None else value,
        parameter_index=index,
        lower_bound=parameter.lower_bound,
        upper_bound=parameter.upper_bound,
    )


def describe(constraints):
    return [(c.label, c.lower_bound, c.upper_bound) for c in constraints]


def feasible(model, constraints, point):
    theta = np.array(point, dtype=float)
    if np.any(theta < model.lower) or np.any(theta > model.upper):
        return False
    for constraint in constraints:
        value = (
            evaluate_ast(constraint.evaluate, theta,
                         parameter_snapshot=model.parameter_info)
            if isinstance(constraint.evaluate, Node)
            else constraint.evaluate(SimpleNamespace(theta=theta))
        )
        if not np.all((value >= constraint.lower_bound)
                      & (value <= constraint.upper_bound)):
            return False
    return True


def equivalent_at(model, original, reduced, points):
    for point in points:
        expected = feasible(model, original, point)
        actual = feasible(model, reduced, point)
        assert actual == expected, (
            f"At theta={point}: expected feasible={expected}, actual={actual}; "
            f"original={describe(original)}, reduced={describe(reduced)}"
        )


def interval_points(*intervals):
    values = {-4.0, -1.0, 0.0, 1.0, 4.0}
    for interval in intervals:
        for boundary in interval:
            if np.isfinite(boundary):
                values.update((boundary, np.nextafter(boundary, -np.inf),
                               np.nextafter(boundary, np.inf)))
    return [(value,) for value in sorted(values)]


def halfplanes(model, constraints):
    """Numerically characterize these test-only affine ASTs, without parsing them.

    Their coefficients are small integers/dyadic fractions, so converting the
    numerical probes to Fraction gives an exact independent polygon oracle.
    """
    rows = []
    for index, (lower, upper) in enumerate(zip(model.lower, model.upper)):
        unit = [Fraction(0), Fraction(0)]
        unit[index] = Fraction(1)
        rows.append((tuple(unit), Fraction(float(upper))))
        rows.append((tuple(-v for v in unit), -Fraction(float(lower))))
    for constraint in constraints:
        values = [Fraction(float(evaluate_ast(
            constraint.evaluate, np.array(point, dtype=float), model.parameter_info
        ))) for point in ((0, 0), (1, 0), (0, 1))]
        offset = values[0]
        coefficients = (values[1]-offset, values[2]-offset)
        if np.isfinite(constraint.upper_bound):
            rows.append((coefficients, Fraction(float(constraint.upper_bound))-offset))
        if np.isfinite(constraint.lower_bound):
            rows.append((tuple(-v for v in coefficients),
                         offset-Fraction(float(constraint.lower_bound))))
    return rows


def polygon_vertices(rows):
    vertices = set()
    for (a, b), (c, d) in combinations(rows, 2):
        determinant = a[0]*c[1]-a[1]*c[0]
        if determinant == 0:
            continue
        point = ((b*c[1]-a[1]*d)/determinant,
                 (a[0]*d-b*c[0])/determinant)
        if all(sum(v*w for v, w in zip(coefficients, point)) <= bound
               for coefficients, bound in rows):
            vertices.add(point)
    return vertices


def object_snapshot(constraints, model):
    """Record public objects by identity; never invoke symbolic __eq__."""
    records, seen = [], set()

    def freeze(value):
        if isinstance(value, (Node, Constraint, ParameterInfo)):
            visit(value)
            return (type(value).__name__, id(value))
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes(), value.flags.writeable)
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return (type(value).__name__, id(value))

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        fields = tuple((key, freeze(value)) for key, value in vars(obj).items())
        records.append((id(obj), fields))

    for constraint in constraints:
        visit(constraint)
    for parameter in model.parameters:
        visit(parameter)
    mapping = tuple((key, freeze(value)) for key, value in model.parameter_info.items())
    return tuple(sorted(records)), mapping, freeze(model.lower), freeze(model.upper)


INTERVAL_PAIRS = [
    ((0.0, np.inf), (0.0, np.inf)),
    ((0.0, np.inf), (1.0, np.inf)),
    ((-np.inf, 3.0), (-np.inf, 2.0)),
    ((0.0, 3.0), (1.0, 2.0)),
    ((1.0, 1.0), (0.0, 2.0)),
    ((-1.0, 2.0), (0.0, 3.0)),
    ((0.0, np.inf), (-np.inf, 2.0)),
    ((-np.inf, np.inf), (0.0, 2.0)),
    ((1.0, np.inf), (-np.inf, 0.0)),
    ((0.0, 0.0), (1.0, 1.0)),
]


@pytest.mark.parametrize("first,second", INTERVAL_PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_interval_intersection_is_preserved(first, second, reverse):
    model = domain((-5.0, 5.0))
    x, = model.parameters
    original = [Constraint(x, *first, label="first"),
                Constraint(x, *second, label="second")]
    if reverse:
        original.reverse()
    reduced = model.reduce(remove_duplicate_constraints, original)
    equivalent_at(model, original, reduced, interval_points(first, second))
    # A containing interval is unnecessary when the stronger interval exists.
    nested = ((first[0] <= second[0] and first[1] >= second[1])
              or (second[0] <= first[0] and second[1] >= first[1]))
    if nested:
        assert len(reduced) == 1, f"Expected one strongest interval, got {describe(reduced)}"


def test_duplicate_same_constraint_object_is_removed():
    model = domain((-2.0, 2.0))
    constraint = Constraint(model.parameters[0], lower_bound=0.5, label="keep")
    reduced = model.reduce(remove_duplicate_constraints, [constraint] * 3)
    assert len(reduced) == 1 and reduced[0] is constraint


@pytest.mark.parametrize("expression", [lambda x: x*x, lambda x: x.sin(), lambda x: x.exp()],
                         ids=["square", "sin", "exp"])
@pytest.mark.parametrize("same_object", [False, True])
def test_duplicate_nonlinear_expression(expression, same_object):
    model = domain((-1.0, 1.0))
    x, = model.parameters
    first = expression(x)
    second = first if same_object else expression(x)
    original = (Constraint(first, upper_bound=2.0, label="a"),
                Constraint(second, upper_bound=2.0, label="b"))
    reduced = model.reduce(remove_duplicate_constraints, original)
    assert len(reduced) == 1, f"Expected one nonlinear duplicate, got {describe(reduced)}"
    equivalent_at(model, original, reduced, [(p,) for p in (-1.0, 0.0, 0.75, 1.0)])


def test_duplicate_same_callable_identity_without_execution():
    def evaluator(context):
        pytest.fail("Duplicate reduction must not execute an arbitrary callable")

    constraints = [Constraint(evaluator, lower_bound=0.0),
                   Constraint(evaluator, lower_bound=1.0)]
    reduced = tuple(remove_duplicate_constraints(constraints))
    assert len(reduced) == 1 and reduced[0] is constraints[1]


def test_duplicate_separate_callables_are_not_assumed_equivalent():
    first = Constraint(lambda ctx: ctx.theta[0]**2, upper_bound=1.0)
    second = Constraint(lambda ctx: ctx.theta[0]**2, upper_bound=1.0)
    reduced = tuple(remove_duplicate_constraints([first, second]))
    assert [id(c) for c in reduced] == [id(first), id(second)]


def test_duplicate_different_parameter_identities_are_preserved():
    first = Parameter(0.0, name="same").variable()
    second = Parameter(0.0, name="same").variable()
    constraints = (Constraint(first**2, upper_bound=1.0),
                   Constraint(second**2, upper_bound=1.0))
    reduced = tuple(remove_duplicate_constraints(constraints))
    assert [id(c) for c in reduced] == [id(c) for c in constraints]


def test_duplicate_different_expressions_are_incomparable():
    model = domain((-2.0, 2.0))
    x, = model.parameters
    constraints = [Constraint(x*x, upper_bound=1.0),
                   Constraint(-x, upper_bound=0.0)]
    reduced = model.reduce(remove_duplicate_constraints, constraints)
    equivalent_at(model, constraints, reduced, [(-2.0,), (-0.5,), (0.5,), (2.0,)])
    assert len(reduced) == 2


@pytest.mark.parametrize("relation", ["gt", "lt"])
def test_duplicate_stronger_strict_interval_preserves_nextafter_boundary(relation):
    model = domain((0.0, 2.0))
    x, = model.parameters
    strict = normalize_to_constraint(x > 1.0 if relation == "gt" else x < 1.0)
    weak = Constraint(x, lower_bound=1.0) if relation == "gt" else Constraint(x, upper_bound=1.0)
    original = [weak, strict]
    reduced = model.reduce(remove_duplicate_constraints, original)
    equivalent_at(model, original, reduced, interval_points((1.0, 1.0)))
    assert len(reduced) == 1


@pytest.mark.parametrize("function", [remove_duplicate_constraints, remove_bound_redundant_constraints,
                                      remove_affine_redundant_constraints])
def test_unconstrained_interval_is_removed(function):
    model = domain((0.0, 1.0))
    original = [Constraint(model.parameters[0])]
    reduced = model.reduce(function, original)
    assert not reduced, f"Expected removal of [-inf, inf], got {describe(reduced)}"


BOUND_CASES = [
    ((0.0, 1.0), (-np.inf, 1.0), True),
    ((0.0, 1.0), (0.0, np.inf), True),
    ((0.0, 1.0), (-1.0, 2.0), True),
    ((0.0, 1.0), (0.0, 1.0), True),
    ((0.0, 1.0), (0.25, np.inf), False),
    ((0.0, 1.0), (-np.inf, 0.75), False),
    ((0.0, 1.0), (0.25, 2.0), False),
    ((0.0, 1.0), (-1.0, 0.75), False),
    ((0.0, 1.0), (0.25, 0.75), False),
    ((0.0, 1.0), (0.5, 0.5), False),
    ((0.5, 0.5), (0.5, 0.5), True),
    ((-np.inf, np.inf), (0.0, np.inf), False),
    ((0.0, np.inf), (0.0, np.inf), True),
    ((-np.inf, 1.0), (-np.inf, 1.0), True),
    ((-np.inf, np.inf), (-np.inf, np.inf), True),
]


@pytest.mark.parametrize("bounds,interval,removed", BOUND_CASES)
@pytest.mark.parametrize("wrapped", [False, True])
def test_bound_direct_variable_interval(bounds, interval, removed, wrapped):
    model = domain(bounds)
    x, = model.parameters
    expression = InputNode(InputNode(x)) if wrapped else x
    original = [Constraint(expression, *interval, label="interval")]
    reduced = model.reduce(remove_bound_redundant_constraints, original)
    assert bool(reduced) is not removed, (
        f"Bounds={bounds}, constraint={interval}: expected removed={removed}, "
        f"got {describe(reduced)}"
    )
    equivalent_at(model, original, reduced, interval_points(bounds, interval))
    if reduced:
        assert reduced[0].label == "interval"


def test_bound_uses_canonical_index_not_mapping_insertion_order():
    model = domain((0.0, 1.0), (10.0, 20.0))
    x, y = model.parameters
    model.parameter_info = dict(reversed(list(model.parameter_info.items())))
    original = [Constraint(InputNode(y), lower_bound=10.0, label="implied y"),
                Constraint(x, upper_bound=0.5, label="needed x")]
    reduced = model.reduce(remove_bound_redundant_constraints, original)
    assert len(reduced) == 1 and reduced[0] is original[1], describe(reduced)


def test_bound_snapshot_membership_and_supplied_bounds_override_live_parameter():
    model = domain((0.0, 1.0))
    x, = model.parameters
    x.fixed()
    x.lower_bound, x.upper_bound = -10.0, 10.0
    original = [Constraint(x, lower_bound=0.0)]
    assert not model.reduce(remove_bound_redundant_constraints, original)


@pytest.mark.parametrize("fixed_value,lower,upper", [(2.0, 1.0, 3.0), (2.0, 2.0, 2.0),
                                                    (2.0, 3.0, np.inf)])
def test_bound_fixed_parameter_does_not_consume_a_theta_dimension(fixed_value, lower, upper):
    model = domain((0.0, 1.0))
    fixed = Parameter(fixed_value, name="fixed")
    model.parameter_info[id(fixed)] = metadata(fixed, None)
    original = [Constraint(InputNode(fixed), lower, upper, label="fixed")]
    reduced = model.reduce(remove_bound_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(0.0,), (0.5,), (1.0,)])


def test_bound_fixed_snapshot_cannot_be_replaced_by_live_variable_state():
    model = domain((0.0, 1.0))
    fixed = Parameter(2.0)
    model.parameter_info[id(fixed)] = metadata(fixed, None)
    fixed.value = 0.0
    fixed.variable()
    original = [Constraint(fixed, upper_bound=1.0)]
    reduced = model.reduce(remove_bound_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(0.0,), (1.0,)])
    assert reduced, "An unsatisfied fixed-snapshot constraint cannot be discarded"


@pytest.mark.parametrize("wrapped", [False, True])
def test_bound_parameter_absent_from_snapshot_is_preserved(wrapped):
    model = domain((0.0, 1.0))
    external = Parameter(0.5).variable(0.0, 1.0)
    expression = InputNode(external) if wrapped else external
    constraint = Constraint(expression, lower_bound=0.0)
    reduced = model.reduce(remove_bound_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


@pytest.mark.parametrize("relation", ["gt", "lt"])
@pytest.mark.parametrize("bound_covers_strict", [False, True])
def test_bound_strict_inequality_keeps_nextafter_distinction(relation, bound_covers_strict):
    boundary = 1.0
    strict = np.nextafter(boundary, np.inf if relation == "gt" else -np.inf)
    bounds = ((strict if bound_covers_strict else boundary, 2.0)
              if relation == "gt"
              else (0.0, strict if bound_covers_strict else boundary))
    model = domain(bounds)
    x, = model.parameters
    constraint = normalize_to_constraint(x > boundary if relation == "gt" else x < boundary)
    actual_threshold = constraint.lower_bound if relation == "gt" else constraint.upper_bound
    assert actual_threshold == strict
    reduced = model.reduce(remove_bound_redundant_constraints, [constraint])
    equivalent_at(model, [constraint], reduced, [(boundary,), (strict,)])
    assert bool(reduced) is not bound_covers_strict, describe(reduced)


@pytest.mark.parametrize("expression", [lambda x: x+0.0, lambda x: 2.0*x,
                                       lambda x: x*x, lambda x: x.sin()])
def test_bound_does_not_reduce_derived_expressions(expression):
    model = domain((0.0, 1.0))
    constraint = Constraint(expression(model.parameters[0]), upper_bound=10.0)
    reduced = model.reduce(remove_bound_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


def test_bound_does_not_execute_callable():
    def evaluator(context):
        pytest.fail("Bound reduction must not invoke an arbitrary callable")

    model = domain((0.0, 1.0))
    constraint = Constraint(evaluator, upper_bound=10.0)
    reduced = model.reduce(remove_bound_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


AFFINE_EXPRESSIONS = [
    (lambda x, y: Literal(3.0), 3.0, 3.0),
    (lambda x, y: x+y, -3.0, 5.0),
    (lambda x, y: x-y, -4.0, 4.0),
    (lambda x, y: -x, -3.0, 2.0),
    (lambda x, y: +x, -2.0, 3.0),
    (lambda x, y: 3.0*x, -6.0, 9.0),
    (lambda x, y: x*3.0, -6.0, 9.0),
    (lambda x, y: x/2.0, -1.0, 1.5),
    (lambda x, y: 2.0-(x+y)/2.0, -0.5, 3.5),
    (lambda x, y: InputNode(x)+InputNode(y), -3.0, 5.0),
]


@pytest.mark.parametrize("expression,minimum,maximum", AFFINE_EXPRESSIONS,
                         ids=["constant", "add", "subtract", "negate", "positive",
                              "left_multiply", "right_multiply", "divide", "derived", "handles"])
@pytest.mark.parametrize("sides", ["lower", "upper", "both"])
def test_affine_exact_extrema_implied_by_parameter_bounds(expression, minimum, maximum, sides):
    model = domain((-2.0, 3.0), (-1.0, 2.0))
    lower = minimum if sides != "upper" else -np.inf
    upper = maximum if sides != "lower" else np.inf
    original = [Constraint(expression(*model.parameters), lower, upper, label="implied")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert not reduced, f"Expected tight implied affine interval removed, got {describe(reduced)}"
    equivalent_at(model, original, reduced, product((-2.0, 0.0, 3.0), (-1.0, 0.0, 2.0)))


@pytest.mark.parametrize("sides", ["lower", "upper", "both"])
def test_affine_slack_extrema_implied_by_parameter_bounds(sides):
    model = domain((-2.0, 3.0), (-1.0, 2.0))
    x, y = model.parameters
    original = [Constraint(x+y, -4.0 if sides != "upper" else -np.inf,
                           6.0 if sides != "lower" else np.inf, label="slack")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert not reduced, f"Expected slack implied interval removed, got {describe(reduced)}"
    equivalent_at(model, original, reduced, product((-2.0, 0.0, 3.0), (-1.0, 0.0, 2.0)))


@pytest.mark.parametrize("expression", [lambda x: 0.0*x, lambda x: x*0.0,
                                       lambda x: x-x], ids=["left_zero", "right_zero", "cancel"])
def test_affine_zero_coefficients_are_constant_equalities(expression):
    model = domain((-5.0, 5.0))
    original = [Constraint(expression(model.parameters[0]), 0.0, 0.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert not reduced, f"Identically zero expression is redundant, got {describe(reduced)}"


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_affine_constant_equality_is_removed(value):
    model = domain((0.0, 1.0))
    assert not model.reduce(remove_affine_redundant_constraints,
                            [Constraint(Literal(value), value, value)])


def test_affine_fixed_parameter_scalar_uses_frozen_snapshot():
    model = domain((0.0, 1.0))
    x, = model.parameters
    factor = Parameter(2.0)
    model.parameter_info[id(factor)] = metadata(factor, None)
    factor.value = 100.0
    original = [Constraint(x*factor+factor, 2.0, 4.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(0.0,), (0.5,), (1.0,)])
    assert not reduced, describe(reduced)


def test_affine_division_by_fixed_snapshot_parameter():
    model = domain((0.0, 4.0))
    divisor = Parameter(2.0)
    model.parameter_info[id(divisor)] = metadata(divisor, None)
    divisor.value = 0.5
    constraint = Constraint(model.parameters[0]/divisor, 0.0, 2.0)
    assert not model.reduce(remove_affine_redundant_constraints, [constraint])


def test_affine_reordered_canonical_indices():
    model = domain((0.0, 1.0), (10.0, 20.0))
    x, y = model.parameters
    model.parameter_info = dict(reversed(list(model.parameter_info.items())))
    constraint = Constraint(2.0*x-y, -20.0, -8.0)
    assert not model.reduce(remove_affine_redundant_constraints, [constraint])


@pytest.mark.parametrize("case", ["multiply", "divide", "canonical_index"])
def test_affine_slack_cases_isolate_snapshot_and_index_semantics(case):
    if case == "canonical_index":
        model = domain((0.0, 1.0), (10.0, 20.0))
        x, y = model.parameters
        model.parameter_info = dict(reversed(list(model.parameter_info.items())))
        constraint = Constraint(2*x-y, -21.0, -7.0)
    else:
        model = domain((0.0, 1.0) if case == "multiply" else (0.0, 4.0))
        x, = model.parameters
        fixed = Parameter(2.0)
        model.parameter_info[id(fixed)] = metadata(fixed, None)
        if case == "multiply":
            fixed.value = 100.0
            constraint = Constraint(x*fixed+fixed, 1.0, 5.0)
        else:
            fixed.value = 0.5
            constraint = Constraint(x/fixed, -1.0, 3.0)
    reduced = model.reduce(remove_affine_redundant_constraints, [constraint])
    assert not reduced, f"Expected snapshot-implied slack interval removed, got {describe(reduced)}"


def test_affine_variable_snapshot_survives_live_fixed_state():
    model = domain((0.0, 2.0))
    x, = model.parameters
    x.fixed()
    constraint = Constraint(x+1.0, upper_bound=2.0)
    reduced = model.reduce(remove_affine_redundant_constraints, [constraint])
    equivalent_at(model, [constraint], reduced, [(0.0,), (1.0,), (1.5,), (2.0,)])
    assert len(reduced) == 1, "Snapshot-variable constraint remains necessary"


def test_affine_single_other_constraint_implies_weaker_constraint():
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    original = [Constraint(x+y, upper_bound=1.0, label="sum"),
                Constraint(x, upper_bound=1.0, label="implied x")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert len(reduced) == 1 and reduced[0].label == "sum", describe(reduced)
    equivalent_at(model, original, reduced, product((0.0, 0.5, 1.0, 2.0), repeat=2))


@pytest.mark.parametrize("lower", [False, True])
def test_affine_multiple_other_constraints_imply_sum(lower):
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    original = ([Constraint(x, lower_bound=0.25, label="x"),
                 Constraint(y, lower_bound=0.5, label="y"),
                 Constraint(x+y, lower_bound=0.75, label="implied sum")]
                if lower else
                [Constraint(x, upper_bound=0.5, label="x"),
                 Constraint(y, upper_bound=0.5, label="y"),
                 Constraint(x+y, upper_bound=1.0, label="implied sum")])
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert [c.label for c in reduced] == ["x", "y"], describe(reduced)
    equivalent_at(model, original, reduced, product((0.0, 0.25, 0.5, 0.75, 1.0, 2.0), repeat=2))


@pytest.mark.parametrize("equivalent,lower,upper", [
    (lambda x, y: 2*x+2*y, -np.inf, 2.0),
    (lambda x, y: y+x, -np.inf, 1.0),
    (lambda x, y: -(x+y), -1.0, np.inf),
], ids=["multiply", "commute", "negate"])
def test_affine_mutually_redundant_representations_keep_one(equivalent, lower, upper):
    model = domain((0.0, 1.0), (0.0, 1.0))
    x, y = model.parameters
    second = equivalent(x, y)
    other = Constraint(second, lower, upper)
    original = [Constraint(x+y, upper_bound=1.0), other]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((0.0, 0.5, 1.0), repeat=2))
    assert len(reduced) == 1, f"Expected exactly one necessary representative, got {describe(reduced)}"


def test_affine_equivalent_equalities_keep_one():
    model = domain((-1.0, 1.0), (-1.0, 1.0))
    x, y = model.parameters
    original = [Constraint(x-y, 0.0, 0.0), Constraint(y-x, 0.0, 0.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((-1.0, 0.0, 1.0), repeat=2))
    assert len(reduced) == 1, describe(reduced)


def test_affine_two_equalities_imply_third_without_losing_rank():
    model = domain((-5.0, 5.0), (-5.0, 5.0))
    x, y = model.parameters
    original = [Constraint(x, 1.0, 1.0), Constraint(y, 2.0, 2.0),
                Constraint(x+y, 3.0, 3.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((0.0, 1.0, 2.0, 3.0), repeat=2))
    assert len(reduced) == 2, describe(reduced)


def test_affine_equality_and_one_sided_constraint_preserve_feasible_segment():
    model = domain((-2.0, 2.0), (-2.0, 2.0))
    x, y = model.parameters
    original = [Constraint(x+y, 1.0, 1.0),
                Constraint(x, lower_bound=0.0), Constraint(y, upper_bound=1.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((-1.0, 0.0, 0.5, 1.0, 2.0), repeat=2))
    assert len(reduced) == 2, describe(reduced)


def test_affine_partial_two_sided_redundancy_keeps_active_side():
    model = domain((0.0, 1.0), (0.0, 1.0))
    x, y = model.parameters
    original = [Constraint(x+y, 0.0, 1.5, label="partial")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((0.0, 0.5, 1.0), repeat=2))
    assert len(reduced) == 1 and reduced[0].label == "partial"


@pytest.mark.parametrize("upper", [0.999, 0.99999999])
def test_affine_almost_implied_constraint_must_not_be_removed(upper):
    model = domain((0.0, 1.0))
    original = [Constraint(model.parameters[0]+0.0, upper_bound=upper)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(upper,), (1.0,)])
    assert len(reduced) == 1, f"x=1 violates upper={upper}, so this constraint is necessary"


@pytest.mark.parametrize("margin", [1e-6, 1e-8, 1e-10, 1e-12])
def test_affine_optical_scale_small_violation_must_not_be_rounded_away(margin):
    model = domain((0.004, 0.010), (0.005, 0.014))
    d1, d2 = model.parameters
    original = [Constraint(d1+d2, upper_bound=0.024-margin)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    # The far corner is inside the parameter box but violates the extra limit.
    assert not feasible(model, original, (0.010, 0.014))
    equivalent_at(model, original, reduced, [(0.004, 0.005), (0.010, 0.014)])
    assert len(reduced) == 1, f"A genuine {margin:g} violation cannot be ignored"


def test_affine_partially_overlapping_intervals_preserve_intersection():
    model = domain((-5.0, 5.0))
    x, = model.parameters
    original = [Constraint(2.0*x, 0.0, 4.0, label="first"),
                Constraint(x+0.0, 1.0, 3.0, label="second")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, interval_points((0.0, 2.0), (1.0, 3.0)))
    assert feasible(model, reduced, (1.5,))
    assert not feasible(model, reduced, (0.5,))
    assert not feasible(model, reduced, (2.5,))


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_affine_implications_are_preserved_across_scales(scale):
    model = domain((0.0, 2.0*scale), (0.0, 2.0*scale))
    x, y = model.parameters
    original = [Constraint(x, upper_bound=0.5*scale, label="x"),
                Constraint(y, upper_bound=0.5*scale, label="y"),
                Constraint(x+y, upper_bound=scale, label="implied sum")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((0.0, 0.5*scale, scale, 2.0*scale), repeat=2))
    assert [c.label for c in reduced] == ["x", "y"], describe(reduced)


@pytest.mark.parametrize("expression", [lambda x, y: x*x, lambda x, y: x*y,
                                       lambda x, y: x.sin(), lambda x, y: x/y,
                                       lambda x, y: x.maximum(y), lambda x, y: x.exp()])
def test_affine_unsupported_nonlinear_expressions_remain(expression):
    model = domain((1.0, 2.0), (1.0, 2.0))
    constraint = Constraint(expression(*model.parameters), upper_bound=10.0)
    reduced = model.reduce(remove_affine_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


def test_affine_symbols_remain_unresolved_external_constraints():
    model = domain((0.0, 1.0))
    constraint = Constraint(model.parameters[0]+Symbol("runtime"), upper_bound=1.0)
    reduced = model.reduce(remove_affine_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


def test_affine_unknown_variable_cannot_be_assumed_constant():
    model = domain((0.0, 1.0))
    external = Parameter(0.0).variable()
    constraint = Constraint(model.parameters[0]+external, upper_bound=1.0)
    reduced = model.reduce(remove_affine_redundant_constraints, [constraint])
    assert len(reduced) == 1 and reduced[0] is constraint


def test_affine_does_not_execute_callable_or_use_it_as_lp_evidence():
    def evaluator(context):
        pytest.fail("Affine reduction must not execute a runtime callable")

    model = domain((0.0, 2.0))
    original = [Constraint(evaluator, upper_bound=0.0, label="runtime"),
                Constraint(model.parameters[0], upper_bound=1.0, label="needed")]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    assert len(reduced) == 2 and reduced[0].evaluate is evaluator
    assert describe(reduced) == describe(original)


def test_affine_unbounded_objective_cannot_prove_redundancy():
    model = domain((-np.inf, np.inf), (-np.inf, np.inf))
    x, y = model.parameters
    original = [Constraint(x-y, upper_bound=1.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1e6, 0.0)])
    assert len(reduced) == 1


def test_affine_unbounded_parameter_domain_can_have_bounded_constraint_optimum():
    model = domain((-np.inf, np.inf), (-np.inf, np.inf))
    x, y = model.parameters
    original = [Constraint(x, upper_bound=1.0), Constraint(y, upper_bound=1.0),
                Constraint(x+y, upper_bound=2.0)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, product((-100.0, 0.0, 1.0, 2.0), repeat=2))
    assert len(reduced) == 2


@pytest.mark.parametrize("constant", [False, True])
def test_affine_infeasible_original_must_not_become_feasible(constant):
    model = domain((-2.0, 2.0))
    x, = model.parameters
    original = ([Constraint(Literal(-1.0), lower_bound=0.0)] if constant else
                [Constraint(x, lower_bound=1.0), Constraint(x, upper_bound=0.0)])
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, [(-2.0,), (-0.5,), (0.0,), (0.5,), (1.0,), (2.0,)])
    assert reduced, "An infeasible original system cannot reduce to an unconstrained system"


@pytest.mark.parametrize("value,lower,upper,removed", [(2.0, 2.0, 2.0, True),
                                                      (2.0, 3.0, np.inf, False)])
def test_affine_constant_system_without_any_variables(value, lower, upper, removed):
    model = domain()
    original = [Constraint(Literal(value), lower, upper)]
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, [()])
    assert bool(reduced) is not removed


@pytest.mark.parametrize("function", [remove_duplicate_constraints, remove_bound_redundant_constraints,
                                      remove_affine_redundant_constraints])
def test_reduction_does_not_mutate_parameters_handles_graphs_constraints_or_bounds(function):
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    handle = InputNode(InputNode(x))
    original = [Constraint(handle+y, upper_bound=2.0, label="sum"),
                Constraint(x, lower_bound=0.0, label="bound"),
                Constraint(x*x, upper_bound=4.0, label="nonlinear")]
    before = object_snapshot(original, model)
    model.reduce(function, original)
    assert object_snapshot(original, model) == before


@pytest.mark.parametrize("function", [remove_bound_redundant_constraints, remove_affine_redundant_constraints])
def test_reducers_accept_read_only_snapshot_mapping_and_bound_arrays(function):
    model = domain((0.0, 1.0))
    model.parameter_info = MappingProxyType(model.parameter_info)
    model.lower.setflags(write=False)
    model.upper.setflags(write=False)
    original = [Constraint(model.parameters[0], upper_bound=0.5, label="necessary")]
    before = object_snapshot(original, model)
    reduced = model.reduce(function, original)
    equivalent_at(model, original, reduced, [(0.0,), (0.5,), (1.0,)])
    assert object_snapshot(original, model) == before


@pytest.mark.parametrize("function", [remove_duplicate_constraints, remove_bound_redundant_constraints,
                                      remove_affine_redundant_constraints])
def test_necessary_constraints_keep_meaning_labels_bounds_and_order(function):
    model = domain((-10.0, 10.0), (-10.0, 10.0))
    x, y = model.parameters
    original = [Constraint(x, lower_bound=0.0, label="first"),
                Constraint(y, upper_bound=0.0, label="second"),
                Constraint(x*x+y*y, upper_bound=9.0, label="third")]
    reduced = model.reduce(function, original)
    assert describe(reduced) == describe(original)
    equivalent_at(model, original, reduced, product((-4.0, -1.0, 0.0, 1.0, 4.0), repeat=2))


@pytest.mark.parametrize("function", [remove_duplicate_constraints, remove_bound_redundant_constraints,
                                      remove_affine_redundant_constraints])
@pytest.mark.parametrize("container", [list, tuple, iter])
def test_reducers_accept_constraint_iterables_and_empty_input(function, container):
    model = domain((0.0, 1.0))
    assert model.reduce(function, container([])) == ()
    constraint = Constraint(model.parameters[0], upper_bound=0.5, label="necessary")
    reduced = model.reduce(function, container([constraint]))
    assert len(reduced) == 1 and describe(reduced) == describe([constraint])
    equivalent_at(model, [constraint], reduced, [(0.0,), (0.5,), (1.0,)])


def fiber_model():
    model = domain((0.004, 0.010), (0.005, 0.014))
    d1, d2 = model.parameters
    d3 = 0.025-d1-d2
    original = [Constraint(d3, lower_bound=0.004, label="output clearance"),
                Constraint(d1, lower_bound=0.0, label="input length"),
                Constraint(d2, lower_bound=0.0, label="middle length"),
                Constraint(d3, lower_bound=0.0, label="output length")]
    return model, original


def fiber_points():
    points = list(product((0.004, 0.007, 0.010), (0.005, 0.011, 0.014)))
    points += [(0.00786, 0.01003), (0.010, 0.011), (0.010, 0.01100001)]
    points += [(np.nextafter(0.010, -np.inf), 0.011)]
    rng = np.random.default_rng(420)
    points += list(rng.uniform([0.004, 0.005], [0.010, 0.014], size=(100, 2)))
    return points


def test_affine_fiber_gap_constraints_reduce_to_mechanical_clearance():
    model, original = fiber_model()
    reduced = model.reduce(remove_affine_redundant_constraints, original)
    equivalent_at(model, original, reduced, fiber_points())
    assert len(reduced) == 1 and reduced[0].label == "output clearance", describe(reduced)


def test_full_pipeline_fiber_gaps_preserves_identical_physical_model():
    model, original = fiber_model()
    # Distinct AST and Constraint objects describing the same mechanical limit.
    d1, d2 = model.parameters
    original.append(Constraint(0.025-d1-d2, lower_bound=0.004, label="duplicate clearance"))
    before = object_snapshot(original, model)
    reduced = original
    for function in (remove_duplicate_constraints, remove_bound_redundant_constraints,
                     remove_affine_redundant_constraints):
        next_reduced = model.reduce(function, reduced)
        equivalent_at(model, original, next_reduced, fiber_points())
        reduced = next_reduced
    assert len(reduced) == 1, f"Expected one clearance constraint, got {describe(reduced)}"
    assert reduced[0].label in {"output clearance", "duplicate clearance"}
    assert object_snapshot(original, model) == before
    assert feasible(model, reduced, (0.00786, 0.01003))
    assert not feasible(model, reduced, (0.010, 0.014))
    # Check a physical value directly, independently of AST arithmetic.
    assert 0.025-0.00786-0.01003 >= 0.004
    assert 0.025-0.010-0.014 < 0.004


def test_full_pipeline_mixed_linear_nonlinear_and_callable_constraints():
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    runtime = lambda ctx: ctx.theta[0] + ctx.theta[1]
    original = [Constraint(x+y, upper_bound=1.5, label="sum"),
                Constraint(x+y, upper_bound=2.0, label="weaker sum"),
                Constraint(InputNode(x), lower_bound=0.0, label="parameter bound"),
                Constraint(x*x+y*y, upper_bound=1.0, label="nonlinear"),
                Constraint(runtime, lower_bound=0.25, label="callable")]
    before = object_snapshot(original, model)
    reduced = original
    for function in (remove_duplicate_constraints, remove_bound_redundant_constraints,
                     remove_affine_redundant_constraints):
        reduced = model.reduce(function, reduced)
        equivalent_at(model, original, reduced,
                      product((0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0), repeat=2))
    assert [c.label for c in reduced] == ["sum", "nonlinear", "callable"]
    assert object_snapshot(original, model) == before


@pytest.mark.parametrize("system", ["triangle", "rectangle", "equality", "diagonal", "overlap"])
def test_full_pipeline_exact_polygon_feasible_region_certificate(system):
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    systems = {
        "triangle": [Constraint(x+y, upper_bound=1.0),
                     Constraint(x, upper_bound=1.0), Constraint(y, upper_bound=1.0)],
        "rectangle": [Constraint(x, upper_bound=0.5),
                      Constraint(y, upper_bound=0.5), Constraint(x+y, upper_bound=1.0)],
        "equality": [Constraint(x+y, 1.0, 1.0),
                     Constraint(x, lower_bound=0.0), Constraint(y, upper_bound=1.0)],
        "diagonal": [Constraint(x-y, upper_bound=0.5),
                     Constraint(y-x, upper_bound=0.5), Constraint(x+y, upper_bound=2.0)],
        "overlap": [Constraint(x, 0.25, 1.5), Constraint(2*x, 1.0, 4.0),
                    Constraint(y, lower_bound=0.5)],
    }
    original = systems[system]
    original_vertices = polygon_vertices(halfplanes(model, original))
    assert original_vertices, "The independent fixture describes a nonempty bounded polytope"
    reduced = original
    for function in (remove_duplicate_constraints, remove_bound_redundant_constraints,
                     remove_affine_redundant_constraints):
        reduced = model.reduce(function, reduced)
        actual_vertices = polygon_vertices(halfplanes(model, reduced))
        assert actual_vertices == original_vertices, (
            f"Expected exact polygon vertices {original_vertices}, actual={actual_vertices}; "
            f"reduced={describe(reduced)}"
        )


@pytest.mark.parametrize("make_binding", [
    pytest.param(lambda p: 1.5, id="scalar"),
    pytest.param(lambda p: np.float64(1.5), id="numpy-scalar"),
    pytest.param(lambda p: Literal(1.5), id="literal"),
    pytest.param(lambda p: p, id="fixed-parameter"),
    pytest.param(lambda p: InputNode(p), id="input-node"),
    pytest.param(lambda p: Symbol("alias"), id="symbol-alias"),
    pytest.param(lambda p: 2 * p / 2, id="affine-expression"),
])
def test_affine_context_resolves_redundant_fixed_index_equality(make_binding):
    model = domain((0.0, 2.0))
    index = Parameter(1.5, name="glass index")
    model.parameter_info[id(index)] = metadata(index, None)
    index.value = 5.0
    index.variable()  # Frozen metadata, not the live state, is authoritative.
    key = ("medium", "n")
    context = MappingProxyType({key: make_binding(index), "alias": index})
    expression = index - Symbol(key)
    constraint = Constraint(expression, 0.0, 0.0, label="continuity")
    before = object_snapshot([constraint], model)

    assert model.reduce(remove_affine_redundant_constraints, [constraint]) == (constraint,)
    reduced = remove_affine_redundant_constraints(
        [constraint], parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context=context,
    )

    assert reduced == ()
    assert object_snapshot([constraint], model) == before
    assert context["alias"] is index


@pytest.mark.parametrize("make_binding", [
    pytest.param(lambda p: p, id="parameter"),
    pytest.param(lambda p: InputNode(p), id="input-node"),
    pytest.param(lambda p: Symbol("alias"), id="symbol-alias"),
])
def test_affine_context_resolves_same_variable_with_canonical_snapshot(make_binding):
    model = domain((0.0, 2.0), (1.0, 2.0))
    index = model.parameters[1]  # Not the first local AST variable.
    index.fixed()
    index.value = 10.0
    constraint = Constraint(index - Symbol("n"), 0.0, 0.0)

    reduced = remove_affine_redundant_constraints(
        [constraint], parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context={"n": make_binding(index), "alias": index},
    )

    assert reduced == ()
    assert model.parameter_info[id(index)].parameter_index == 1
    assert index.value == 10.0
    assert not index.is_variable


def test_affine_context_keeps_independent_index_continuity_equality():
    model = domain((1.0, 2.0), (1.0, 2.0))
    upstream, downstream = model.parameters
    constraint = Constraint(upstream - Symbol("n"), 0.0, 0.0, label="physical continuity")

    reduced = remove_affine_redundant_constraints(
        [constraint], parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context={"n": downstream},
    )

    assert len(reduced) == 1 and reduced[0] is constraint
    assert evaluate_ast(reduced[0].evaluate, np.array([1.0, 2.0]),
                        model.parameter_info, bindings={"n": 2.0}) == -1.0


@pytest.mark.parametrize("make_context", [
    pytest.param(lambda p: None, id="no-context"),
    pytest.param(lambda p: {}, id="missing-key"),
    pytest.param(lambda p: {"n": Symbol("missing")}, id="unresolved-alias"),
    pytest.param(lambda p: {"n": Symbol("alias"), "alias": Symbol("n")}, id="cycle"),
    pytest.param(lambda p: {"n": p ** 2}, id="nonlinear"),
    pytest.param(lambda p: {"n": object()}, id="unsupported"),
    pytest.param(lambda p: {"n": "1"}, id="not-numeric"),
    pytest.param(lambda p: {"n": np.inf}, id="infinite"),
])
def test_affine_context_preserves_unprovable_symbol_constraints(make_context):
    model = domain((1.0, 2.0))
    parameter, = model.parameters
    constraint = Constraint(parameter - Symbol("n"), 0.0, 0.0, label="unresolved")

    reduced = remove_affine_redundant_constraints(
        [constraint], parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context=make_context(parameter),
    )

    assert len(reduced) == 1 and reduced[0] is constraint


def test_affine_context_preserves_conflicting_fixed_indices():
    model = domain((0.0, 2.0))
    index = Parameter(1.5)
    model.parameter_info[id(index)] = metadata(index, None)
    constraint = Constraint(index - Symbol("n"), 0.0, 0.0, label="contradiction")

    reduced = remove_affine_redundant_constraints(
        [constraint], parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context={"n": 1.6},
    )

    assert len(reduced) == 1 and reduced[0] is constraint
    assert evaluate_ast(constraint.evaluate, np.array([1.0]),
                        model.parameter_info, bindings={"n": 1.6}) != 0.0


def test_affine_context_preserves_inequality_implications_and_bounds():
    model = domain((0.0, 2.0), (0.0, 2.0))
    x, y = model.parameters
    symbol = Symbol("sum")
    necessary = Constraint(symbol, upper_bound=1.0, label="sum limit")
    constraints = [
        Constraint(Symbol("x"), 0.0, 2.0, label="box redundant"),
        necessary,
        Constraint(2 * symbol, upper_bound=2.0, label="equivalent sum limit"),
    ]

    reduced = remove_affine_redundant_constraints(
        constraints, parameter_info=model.parameter_info,
        parameter_lower=model.lower, parameter_upper=model.upper,
        compile_context={"x": x, "sum": x + y},
    )

    assert len(reduced) == 1
    assert any(reduced[0] is item for item in constraints[1:])
    for a, b in product((0.0, 0.5, 1.0, 2.0), repeat=2):
        value = evaluate_ast(reduced[0].evaluate, np.array([a, b]),
                             model.parameter_info, bindings={"sum": a + b})
        assert bool(reduced[0].lower_bound <= value <= reduced[0].upper_bound) == (a + b <= 1.0)
