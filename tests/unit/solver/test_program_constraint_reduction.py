"""Constraint reduction preserves the feasible set of a full compiled program."""

from itertools import product

import numpy as np
import pytest

from diffractix.graph import Literal, Parameter, Symbol, precompile, remap_parameters
from diffractix.solver.constraint import Constraint, ConstraintDescriptor, prepare_constraints, reduce_constraints


def make_problem(*constraints):
    program = precompile(tuple(item.evaluate for item in constraints))
    descriptors = tuple(
        ConstraintDescriptor(3 * index + 2, item.lower_bound, item.upper_bound, item.label, str(item))
        for index, item in enumerate(constraints)
    )
    return program, descriptors


def reduce(program, descriptors, lower=None, upper=None):
    size = len(program.parameter_indices)
    return reduce_constraints(
        program, descriptors,
        parameter_lower=[-np.inf] * size if lower is None else lower,
        parameter_upper=[np.inf] * size if upper is None else upper,
    )


def assert_same_feasible_points(program, original, indices, active, points, bindings=None):
    assert len(indices) == len(active)
    assert tuple(indices) == tuple(sorted(set(indices)))
    for theta in points:
        values = program.evaluate(theta, {} if bindings is None else bindings)
        expected = all(np.all((item.lower_bound <= value) & (value <= item.upper_bound))
                       for item, value in zip(original, values, strict=True))
        actual = all(np.all((item.lower_bound <= values[index]) & (values[index] <= item.upper_bound))
                     for index, item in zip(indices, active, strict=True))
        assert actual == expected, f"Feasibility changed at theta={theta}"


def test_duplicate_nonlinear_outputs_intersect_bounds_and_keep_metadata():
    x = Parameter(1).variable()
    program, descriptors = make_problem(
        Constraint(x * x, lower_bound=0, upper_bound=4, label="first"),
        Constraint(x * x, lower_bound=1, upper_bound=9, label="second"),
    )
    before = (program._instructions, program._outputs, dict(program.parameter_indices))
    indices, active = reduce(program, descriptors)

    assert len(indices) == 1
    assert active[0].lower_bound == 1
    assert active[0].upper_bound == 4
    source = descriptors[indices[0]]
    assert (active[0].position, active[0].label, active[0].description) == (
        source.position, source.label, source.description,
    )
    assert [(item.lower_bound, item.upper_bound) for item in descriptors] == [(0, 4), (1, 9)]
    assert before == (program._instructions, program._outputs, program.parameter_indices)
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-3, -2, -1, 0, 1, 2, 3)])


def test_duplicate_runtime_array_outputs_reduce_as_one_componentwise_constraint():
    symbol = Symbol("radius")
    program, descriptors = make_problem(
        Constraint(symbol + 1, upper_bound=5),
        Constraint(Symbol("radius") + 1, upper_bound=3),
    )
    indices, active = reduce(program, descriptors)
    assert len(indices) == 1
    assert active[0].upper_bound == 3
    for values in (np.array([0.0, 1.0]), np.array([1.0, 3.0])):
        assert_same_feasible_points(program, descriptors, indices, active, [()], {"radius": values})


def test_unconstrained_output_is_removed_but_distinct_nonlinear_constraints_remain():
    x = Parameter(1).variable()
    program, descriptors = make_problem(
        Constraint(x * x),
        Constraint(x * x, upper_bound=4, label="square"),
        Constraint(abs(x), upper_bound=1, label="absolute"),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (1, 2)
    assert [item.label for item in active] == ["square", "absolute"]


def test_duplicate_equalities_retain_one_equality():
    x = Parameter(1).variable()
    program, descriptors = make_problem(Constraint(x * x, 1, 1), Constraint(x * x, 1, 1))
    indices, active = reduce(program, descriptors)
    assert len(indices) == 1
    assert active[0].lower_bound == active[0].upper_bound == 1


def test_conflicting_duplicate_intervals_raise():
    x = Parameter(1).variable()
    program, descriptors = make_problem(Constraint(x * x, upper_bound=1), Constraint(x * x, lower_bound=2))
    with pytest.raises(ValueError):
        reduce(program, descriptors)


@pytest.mark.parametrize("expression,lower_bound,upper_bound", [
    (lambda x: x, -1, 2),
    (lambda x: 2 * x + 3, 1, 7),
    (lambda x: 3 - 2 * x, -1, 5),
    (lambda x: -(x / 2), -1, 0.5),
    (lambda x: +x, -1, 2),
    (lambda x: x ** 1, -1, 2),
    (lambda x: x ** 0, 1, 1),
    (lambda x: x - x, 0, 0),
])
def test_affine_box_extrema_include_exact_boundaries(expression, lower_bound, upper_bound):
    x = Parameter(1).variable()
    program, descriptors = make_problem(Constraint(expression(x), lower_bound, upper_bound))
    assert reduce(program, descriptors, [-1], [2]) == ((), ())


@pytest.mark.parametrize("bounds,expected", [
    ((0, 0.75), (-np.inf, 0.75)),
    ((0.25, 1), (0.25, np.inf)),
    ((0.25, 0.75), (0.25, 0.75)),
    ((0.5, 0.5), (0.5, 0.5)),
])
def test_affine_box_removes_only_implied_constraint_sides(bounds, expected):
    x = Parameter(0.5).variable()
    program, descriptors = make_problem(Constraint(x, *bounds, label="bounded"))
    indices, active = reduce(program, descriptors, [0], [1])
    assert indices == (0,)
    assert (active[0].lower_bound, active[0].upper_bound) == expected
    assert active[0].label == "bounded"
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (0, 0.25, 0.5, 0.75, 1)])


def test_affine_box_uses_canonical_indices_and_frozen_fixed_values():
    x, y = Parameter(1).variable(), Parameter(2).variable()
    fixed = Parameter(3)
    program, descriptors = make_problem(Constraint(x + 2 * y + fixed, 3, 8))
    program = remap_parameters(program, {id(y): 0, id(x): 1})
    fixed.value = 100
    assert program.evaluate([2, 1]) == (8,)
    assert reduce(program, descriptors, [0, 0], [2, 1]) == ((), ())


def test_infinite_parameter_bounds_do_not_imply_finite_constraint():
    x, y = Parameter(1).variable(), Parameter(2).variable()
    program, descriptors = make_problem(Constraint(x + 0 * y, upper_bound=2))
    indices, active = reduce(program, descriptors, [-np.inf, -np.inf], [np.inf, np.inf])
    assert indices == (0,)
    assert active[0].upper_bound == 2


def test_zero_coefficient_does_not_poison_box_extrema_with_infinity():
    x, y = Parameter(1).variable(), Parameter(2).variable()
    program, descriptors = make_problem(Constraint(x + 0 * y, 0, 1))
    assert reduce(program, descriptors, [0, -np.inf], [1, np.inf]) == ((), ())


@pytest.mark.parametrize("value,bounds", [(3, (0, 4)), (0, (0, 0))])
def test_satisfied_constant_outputs_are_removed(value, bounds):
    program, descriptors = make_problem(Constraint(Literal(value), *bounds))
    assert reduce(program, descriptors) == ((), ())


def test_violated_constant_output_raises():
    program, descriptors = make_problem(Constraint(Literal(3), upper_bound=2))
    with pytest.raises(ValueError):
        reduce(program, descriptors)


def test_affine_implication_from_multiple_other_constraints():
    x, y = Parameter(0).variable(), Parameter(0).variable()
    program, descriptors = make_problem(
        Constraint(x, upper_bound=1, label="x"),
        Constraint(y, upper_bound=2, label="y"),
        Constraint(x + y, upper_bound=3, label="sum"),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert [item.label for item in active] == ["x", "y"]
    assert_same_feasible_points(program, descriptors, indices, active, product((-1, 0, 1, 2), (-1, 0, 2, 3)))


def test_affine_implication_can_remove_only_one_side():
    x, y = Parameter(0).variable(), Parameter(0).variable()
    program, descriptors = make_problem(
        Constraint(x, upper_bound=1), Constraint(y, upper_bound=2),
        Constraint(x + y, lower_bound=0, upper_bound=3, label="sum"),
    )
    indices, active = reduce(program, descriptors)
    remaining = dict(zip(indices, active, strict=True))
    assert 2 in remaining
    assert remaining[2].lower_bound == 0
    assert remaining[2].upper_bound == np.inf
    assert_same_feasible_points(program, descriptors, indices, active, product((-2, 0, 1, 2), (-2, 0, 2, 3)))


def test_equivalent_affine_forms_cannot_both_disappear():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x + x, upper_bound=2), Constraint(2 * x, upper_bound=2))
    indices, active = reduce(program, descriptors)
    assert len(indices) == 1
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-1, 0, 1, 1.5)])


def test_dependent_continuity_equalities_keep_an_independent_basis():
    first, second, third = (Parameter(1).variable() for _ in range(3))
    program, descriptors = make_problem(
        Constraint(first - second, 0, 0),
        Constraint(second - third, 0, 0),
        Constraint(first - third, 0, 0),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert all(item.lower_bound == item.upper_bound == 0 for item in active)
    assert_same_feasible_points(program, descriptors, indices, active, product((0, 1, 2), repeat=3))


def test_independent_continuity_equality_remains():
    first, second = Parameter(1).variable(), Parameter(1).variable()
    program, descriptors = make_problem(Constraint(first - second, 0, 0))
    indices, active = reduce(program, descriptors)
    assert indices == (0,)
    assert active == descriptors


def test_inconsistent_affine_constraints_are_reported():
    x = Parameter(1).variable()
    program, descriptors = make_problem(Constraint(x, lower_bound=2), Constraint(2 * x, upper_bound=2))
    with pytest.raises(ValueError):
        reduce(program, descriptors)


def test_runtime_symbols_are_not_treated_as_independent_affine_theta_variables():
    x = Parameter(0).variable()
    runtime = Symbol("beam_width")
    program, descriptors = make_problem(
        Constraint(runtime + x, upper_bound=1, label="runtime"),
        Constraint(x ** 2, upper_bound=0.5, label="nonlinear"),
    )
    indices, active = reduce(program, descriptors, [0], [1])
    assert indices == (0, 1)
    assert [item.label for item in active] == ["runtime", "nonlinear"]


def test_full_pipeline_preserves_fiber_gap_requirement_at_optical_scale():
    first, second = Parameter(0.008).variable(), Parameter(0.01).variable()
    gap = 0.025 - first - second
    program, descriptors = make_problem(
        Constraint(first, lower_bound=0), Constraint(second, lower_bound=0),
        Constraint(gap, lower_bound=0),
        Constraint(0.025 - first - second, lower_bound=0.004, label="clearance"),
    )
    lower, upper = [0.004, 0.005], [0.01, 0.014]
    indices, active = reduce(program, descriptors, lower, upper)
    assert len(indices) == 1
    assert active[0].lower_bound == 0.004
    assert active[0].label in (descriptors[index].label for index in indices)
    assert_same_feasible_points(program, descriptors, indices, active,
                               product((0.004, 0.008, 0.01), (0.005, 0.01, 0.014)))


def test_changing_parameter_bounds_restores_a_previously_redundant_constraint():
    x = Parameter(0.5).variable()
    program, descriptors = make_problem(Constraint(x, upper_bound=1))
    assert reduce(program, descriptors, [0], [0.75]) == ((), ())
    indices, active = reduce(program, descriptors, [0], [2])
    assert indices == (0,)
    assert active == descriptors


def test_constraint_one_float_tighter_than_box_boundary_is_retained():
    x = Parameter(0.5).variable()
    tighter = np.nextafter(1.0, -np.inf)
    program, descriptors = make_problem(Constraint(x, upper_bound=tighter))
    indices, active = reduce(program, descriptors, [0], [1])
    assert indices == (0,)
    assert active[0].upper_bound == tighter
    assert_same_feasible_points(program, descriptors, indices, active, [(tighter,), (1.0,)])


@pytest.mark.parametrize("scale", [1e-3, 1, 1e3])
def test_affine_implication_is_consistent_across_numerical_scales(scale):
    x, y = Parameter(0).variable(), Parameter(0).variable()
    program, descriptors = make_problem(
        Constraint(x, upper_bound=scale),
        Constraint(y, upper_bound=2 * scale),
        Constraint(x + y, upper_bound=4 * scale),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert_same_feasible_points(program, descriptors, indices, active,
                               product((-scale, 0, scale, 2 * scale), (-scale, 0, 2 * scale, 3 * scale)))


def test_structural_matching_preserves_noncommutative_operand_order():
    first, second = Symbol("first"), Symbol("second")
    program, descriptors = make_problem(
        Constraint(first - second, upper_bound=1),
        Constraint(second - first, upper_bound=1),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert_same_feasible_points(program, descriptors, indices, active, [()], {"first": 0, "second": 2})


def test_empty_program_and_descriptor_count_validation():
    assert reduce(precompile(()), ()) == ((), ())
    x = Parameter(0).variable()
    program = precompile((x,))
    with pytest.raises(ValueError, match="descriptor"):
        reduce(program, ())


def test_equal_parameter_bounds_remove_only_constraints_true_at_that_value():
    x = Parameter(2).variable()
    program, descriptors = make_problem(Constraint(3 * x, 6, 6))
    assert reduce(program, descriptors, [2], [2]) == ((), ())
    program, descriptors = make_problem(Constraint(x, lower_bound=3))
    with pytest.raises(ValueError):
        reduce(program, descriptors, [2], [2])


def test_contradictory_dependent_equalities_raise():
    first, second, third = (Parameter(1).variable() for _ in range(3))
    program, descriptors = make_problem(
        Constraint(first - second, 0, 0),
        Constraint(second - third, 0, 0),
        Constraint(first - third, 1, 1),
    )
    with pytest.raises(ValueError):
        reduce(program, descriptors)


@pytest.mark.parametrize("lower,upper", [
    ([], [1]),
    ([0], []),
    ([[0]], [[1]]),
    ([np.nan], [1]),
    ([0], [np.nan]),
    ([2], [1]),
    ([np.inf], [np.inf]),
    ([-np.inf], [-np.inf]),
])
def test_invalid_parameter_bound_shapes_and_values_raise(lower, upper):
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x, upper_bound=1))
    with pytest.raises(ValueError):
        reduce(program, descriptors, lower, upper)


def test_tiny_coefficient_on_unbounded_variable_is_feasible_and_necessary():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(1e-12 * x, lower_bound=1))
    indices, active = reduce(program, descriptors)
    assert indices == (0,)
    assert active == descriptors
    assert_same_feasible_points(program, descriptors, indices, active, [(0,), (1e12,), (2e12,)])


def test_structural_matching_preserves_signed_zero_constants():
    runtime = Symbol("runtime")
    program, descriptors = make_problem(
        Constraint(runtime + 0.0, upper_bound=1),
        Constraint(runtime + -0.0, upper_bound=1),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert active == descriptors


@pytest.mark.parametrize("expression,lower,upper,expected_bounds", [
    (lambda x: x ** 2, [1], [3], (1, 9)),
    (lambda x: x ** 2, [-3], [-1], (1, 9)),
    (lambda x: x ** 2, [-2], [3], (0, 9)),
    (lambda x: abs(x), [1], [3], (1, 3)),
    (lambda x: abs(x), [-3], [-1], (1, 3)),
    (lambda x: abs(x), [-2], [3], (0, 3)),
    (lambda x: abs(x) + 2, [-2], [3], (2, 5)),
    (lambda x: -(x ** 2), [-2], [3], (-9, 0)),
])
def test_interval_bounds_remove_nonlinear_constraints_at_exact_extrema(expression, lower, upper, expected_bounds):
    x = Parameter(1).variable()
    program, descriptors = make_problem(Constraint(expression(x), *expected_bounds))
    assert reduce(program, descriptors, lower, upper) == ((), ())


@pytest.mark.parametrize("expression,expected_bounds", [
    (lambda x, y: x * y, (-6, 6)),
    (lambda x, y: x.minimum(y), (-2, 2)),
    (lambda x, y: x.maximum(y), (-2, 3)),
])
def test_interval_bounds_for_two_variable_operations(expression, expected_bounds):
    x, y = Parameter(1).variable(), Parameter(1).variable()
    program, descriptors = make_problem(Constraint(expression(x, y), *expected_bounds))
    assert reduce(program, descriptors, [-2, -2], [3, 2]) == ((), ())


def test_interval_division_by_strictly_negative_interval():
    x, y = Parameter(1).variable(), Parameter(-1).variable()
    program, descriptors = make_problem(Constraint(x / y, -2, -0.25))
    assert reduce(program, descriptors, [1, -4], [2, -1]) == ((), ())


@pytest.mark.parametrize("denominator_bounds", [(0, 2), (-2, 0), (-2, 2)])
def test_interval_division_touching_or_crossing_zero_remains_unknown(denominator_bounds):
    x, y = Parameter(1).variable(), Parameter(1).variable()
    program, descriptors = make_problem(Constraint(x / y, upper_bound=100))
    indices, active = reduce(program, descriptors, [1, denominator_bounds[0]], [2, denominator_bounds[1]])
    assert indices == (0,)
    assert active == descriptors


def test_interval_reduction_removes_only_the_proven_side():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x ** 2, -1, 0.5, label="square"))
    indices, active = reduce(program, descriptors, [-1], [1])
    assert indices == (0,)
    assert active[0].lower_bound == -np.inf
    assert active[0].upper_bound == 0.5
    assert active[0].label == "square"
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-1, -0.5, 0, 0.5, 1)])


def test_interval_reduction_preserves_an_equality_when_only_one_side_is_implied():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x ** 2, 0, 0))
    indices, active = reduce(program, descriptors, [-1], [1])
    assert indices == (0,)
    assert active == descriptors


@pytest.mark.parametrize("expression,bounds", [
    (lambda x: x ** 2, (-np.inf, -1)),
    (lambda x: abs(x), (3, np.inf)),
    (lambda x: -(x ** 2), (1, np.inf)),
])
def test_interval_range_disjoint_from_constraint_is_infeasible(expression, bounds):
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(expression(x), *bounds))
    with pytest.raises(ValueError):
        reduce(program, descriptors, [-2], [2])


def test_interval_analysis_does_not_assume_independently_bounded_occurrences_are_correlated():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x * x, lower_bound=0))
    indices, active = reduce(program, descriptors, [-1], [1])
    assert indices == (0,)
    assert active == descriptors


def test_interval_analysis_leaves_unsupported_nonlinear_functions_active():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x.sin(), -1, 1))
    indices, active = reduce(program, descriptors, [-1], [1])
    assert indices == (0,)
    assert active == descriptors


def test_interval_reduction_is_recomputed_when_bounds_change():
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(abs(x), upper_bound=1))
    assert reduce(program, descriptors, [-1], [1]) == ((), ())
    indices, active = reduce(program, descriptors, [-2], [2])
    assert indices == (0,)
    assert active == descriptors


def test_one_ulp_coefficient_difference_preserves_independent_equalities():
    x, y = Parameter(1).variable(), Parameter(0).variable()
    different = np.nextafter(1.0, np.inf)
    program, descriptors = make_problem(
        Constraint(x + y, 1, 1),
        Constraint(x + different * y, 1, 1),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert active == descriptors


def test_array_literal_duplicates_intersect_bounds_without_mutating_the_program():
    x = Parameter(0).variable()
    first, second = np.array([1., 2.]), np.array([1., 2.])
    program, descriptors = make_problem(
        Constraint(x + first, lower_bound=0, upper_bound=4, label="first"),
        Constraint(x + second, lower_bound=1, upper_bound=5, label="second"),
    )
    instructions, outputs = program._instructions, program._outputs
    indices, active = reduce(program, descriptors)
    assert indices == (0,)
    assert (active[0].lower_bound, active[0].upper_bound) == (1, 4)
    assert active[0].label == "first"
    assert descriptors[0].lower_bound == 0
    assert program._instructions is instructions
    assert program._outputs is outputs
    np.testing.assert_array_equal(first, second)
    assert first.flags.writeable and second.flags.writeable
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-2, -1, 0, 1, 2, 3, 4)])


@pytest.mark.parametrize("second", [
    np.array([2., 1.], dtype=np.float32),
    np.array([[1., 2.]], dtype=np.float32),
    np.array([1., 2.], dtype=np.float64),
])
def test_array_duplicate_keys_distinguish_values_shapes_and_dtypes(second):
    x = Parameter(0).variable()
    program, descriptors = make_problem(
        Constraint(x + np.array([1., 2.], dtype=np.float32), upper_bound=3),
        Constraint(x + second, upper_bound=3),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert active == descriptors
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-1, 0, 1, 2)])


def test_array_constants_preserve_signed_zero_in_structural_keys():
    runtime = Symbol("runtime")
    program, descriptors = make_problem(
        Constraint(runtime + np.array([0., -0.]), upper_bound=1),
        Constraint(runtime + np.array([-0., 0.]), upper_bound=1),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    assert active == descriptors


@pytest.mark.parametrize("coefficients", [np.array([1.]), np.array([1., -1.]), np.array([[1., 2.]])])
def test_array_expressions_are_conservatively_excluded_from_scalar_affine_proofs(coefficients):
    x = Parameter(0).variable()
    program, descriptors = make_problem(Constraint(x * coefficients, -1, 1))
    indices, active = reduce(program, descriptors, [-1], [1])
    assert indices == (0,)
    assert active == descriptors
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-1, -0.5, 0, 0.5, 1)])


def test_vector_equality_is_not_discarded_as_a_scalar_equality_dependency():
    x = Parameter(1).variable()
    program, descriptors = make_problem(
        Constraint(x, 1, 1), Constraint(x * np.array([1., 2.]), 1, 1, label="vector"),
    )
    indices, active = reduce(program, descriptors, [0], [2])
    assert indices == (0, 1)
    assert active[-1].label == "vector"
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (0, 0.5, 1, 2)])


def test_known_empty_array_outputs_impose_no_constraints():
    x = Parameter(0).variable()
    empty = np.empty((0, 2))
    program, descriptors = make_problem(
        Constraint(x + empty, upper_bound=0),
        Constraint(x + empty, lower_bound=1),
        Constraint(x, upper_bound=2),
    )
    indices, active = reduce(program, descriptors)
    assert indices == (2,)
    assert_same_feasible_points(program, descriptors, indices, active, [(value,) for value in (-1, 0, 1, 2, 3)])


def test_conflicting_bounds_on_unknown_runtime_shapes_preserve_empty_array_semantics():
    runtime = Symbol("runtime")
    program, descriptors = make_problem(Constraint(runtime, upper_bound=0), Constraint(runtime, lower_bound=1))
    indices, active = reduce(program, descriptors)
    assert indices == (0, 1)
    for value in (np.array([]), np.array([0.]), np.array([1., 2.])):
        assert_same_feasible_points(program, descriptors, indices, active, [()], {"runtime": value})


@pytest.mark.parametrize("method", ["__le__", "__lt__", "__ge__", "__gt__", "__eq__"])
@pytest.mark.parametrize("array_on_left", [False, True])
def test_comparisons_with_array_literals_normalize_to_componentwise_residuals(method, array_on_left):
    x = Parameter(0).variable()
    array = Literal(np.array([0., 1.]))
    left, right = (array, x) if array_on_left else (x, array)
    comparison = getattr(left, method)(right)
    program, descriptors, callables = prepare_constraints((comparison,), tracer=None, bindings={})
    indices, active = reduce(program, descriptors)
    assert indices == (0,)
    assert callables == ()
    operation = {"__le__": np.less_equal, "__lt__": np.less, "__ge__": np.greater_equal,
                 "__gt__": np.greater, "__eq__": np.equal}[method]
    for value in (-1., 0., 1., 2.):
        # Bounds are host-side optimizer metadata, including nextafter(0) bounds.
        actual = np.asarray(program.evaluate([value])[0])
        expected = operation(array.value, value) if array_on_left else operation(value, array.value)
        assert np.all((actual >= active[0].lower_bound) & (actual <= active[0].upper_bound)) == np.all(expected)


def test_constant_array_constraints_are_checked_componentwise_during_preparation():
    satisfied = Constraint(Literal(np.array([1., 2.])), 0, 3)
    program, descriptors, callables = prepare_constraints((satisfied,), tracer=None, bindings={})
    assert program.evaluate([]) == ()
    assert descriptors == callables == ()
    for values, message in (([1., 4.], "unsatisfiable"), ([1., np.nan], "non-finite")):
        with pytest.raises(ValueError, match=message):
            prepare_constraints((Constraint(Literal(np.array(values)), upper_bound=3),), tracer=None, bindings={})


@pytest.mark.parametrize("constraint_factory, lower, upper", [
    pytest.param(lambda x: Constraint(x), [-1], [1], id="unbounded"),
    pytest.param(lambda x: Constraint(x, 0, 1), [0], [1], id="box-implied"),
    pytest.param(lambda x: Constraint(x ** 2, 0, 1), [-1], [1], id="interval-implied"),
])
def test_constraints_implied_by_parameter_bounds_are_all_removed(constraint_factory, lower, upper):
    x = Parameter(0).variable()
    program, descriptors = make_problem(constraint_factory(x))
    assert reduce(program, descriptors, lower, upper) == ((), ())


def test_reduced_polygon_has_same_independent_halfplane_feasible_region():
    # Port of the legacy AST reducer's geometric regression, using Program outputs.
    x, y = Parameter(0.5).variable(), Parameter(0.5).variable()
    program, descriptors, _ = prepare_constraints((
        x >= 0, y >= 0, x + y <= 2,
        2 * x + 2 * y <= 4, x <= 3,
    ), tracer=None, bindings={})
    indices, active = reduce_constraints(
        program, descriptors, parameter_lower=[-10, -10], parameter_upper=[10, 10],
    )
    assert len(indices) < len(descriptors)
    mapping = program.parameter_indices
    for first in (-1., 0., 0.5, 1., 2., 2.5, 3.):
        for second in (-1., 0., 0.5, 1., 2., 2.5, 3.):
            theta = [0., 0.]
            theta[mapping[id(x)]], theta[mapping[id(y)]] = first, second
            values = program.evaluate(theta)
            actual = all(d.lower_bound <= values[i] <= d.upper_bound for i, d in zip(indices, active))
            expected = first >= 0 and second >= 0 and first + second <= 2
            assert actual == expected, (first, second)
