"""Optimization equivalence, lifetime safety and numerical corner cases."""
import numpy as np
import pytest

from diffractix.backend import grad, hessian, jit, xp
from diffractix.graph import (
    BinaryInstruction, Constant, Input, Literal, Parameter, Program, Register,
    Symbol, UnaryInstruction, compile_graph, optimize, precompile,
)
from diffractix.graph.ops import Op


def test_common_subexpressions_are_shared_without_merging_distinct_inputs():
    x, y = Parameter(1.0).variable(), Parameter(1.0).variable()
    roots = (x + 1, x + 1, y + 1)
    raw = precompile(roots)
    optimized = optimize(raw)
    assert len(raw._instructions) == 3
    assert len(optimized._instructions) == 2
    assert optimized._outputs[0] == optimized._outputs[1]
    np.testing.assert_array_equal(optimized.evaluate([2.0, 4.0]), [3.0, 3.0, 5.0])
    assert len(raw._instructions) == 3
    assert raw.parameter_indices == optimized.parameter_indices


def test_constant_replacements_propagate_into_outputs_and_following_instructions():
    x = Parameter(1.0).variable()
    raw = precompile((+Literal(2), (+Literal(2)) * x))
    optimized = optimize(raw)
    assert len(optimized._instructions) < len(raw._instructions)
    assert isinstance(optimized._outputs[0], Constant)
    np.testing.assert_array_equal(optimized.evaluate([3.0]), [2, 6])


def test_dead_instructions_removed_and_input_program_not_mutated():
    raw = Program(
        parameter_indices={123: 0}, runtime_symbols={},
        _instructions=(
            BinaryInstruction(Op.ADD, 0, Input(0), Constant(1)),
            BinaryInstruction(Op.MUL, 1, Register(0), Constant(2)),
            UnaryInstruction(Op.NEG, 2, Input(0)),
        ),
        _outputs=(Register(1), Input(0), Constant(3), Register(1)),
        _register_count=3,
    )
    before = (dict(raw.parameter_indices), raw._instructions, raw._outputs, raw._register_count)
    optimized = optimize(raw)
    assert optimized is not raw
    assert len(optimized._instructions) == 2
    assert optimized._register_count < raw._register_count
    assert (raw.parameter_indices, raw._instructions, raw._outputs, raw._register_count) == before
    np.testing.assert_array_equal(optimized.evaluate([4.0]), [10.0, 4.0, 3, 10.0])


def test_dead_program_can_still_return_only_inputs_constants_and_duplicate_outputs():
    raw = Program(
        {10: 0}, {}, (UnaryInstruction(Op.NEG, 0, Input(0)),),
        (Input(0), Constant(7), Input(0)), 1,
    )
    optimized = optimize(raw)
    assert optimized._instructions == ()
    assert optimized._register_count == 0
    assert optimized.evaluate([2.0]) == (2.0, 7, 2.0)


def test_register_lifetimes_preserve_repeated_sources_and_earlier_output_registers():
    x = Parameter(1.0).variable()
    first = x + 1
    square = first * first
    branch = square - first
    final = branch / (x + 2)
    roots = (first, final, square, first, -branch)
    raw = precompile(roots)
    optimized = optimize(raw)
    assert optimized._register_count < raw._register_count
    for point in (-0.5, 0.0, 3.0):
        expected = (point + 1, ((point + 1)**2 - point - 1)/(point + 2),
                    (point + 1)**2, point + 1, -((point + 1)**2 - point - 1))
        np.testing.assert_allclose(optimized.evaluate([point]), expected)
        np.testing.assert_allclose(raw.evaluate([point]), expected)


def test_long_chain_reuses_registers_and_preserves_noncommutative_order():
    x, y = Parameter(1.0).variable(), Parameter(2.0).variable()
    root = ((((x - y) / (y + 2))**3 + 5) * 2 - 1)
    raw, optimized = precompile((root,)), compile_graph((root,))
    assert optimized._register_count < raw._register_count
    assert optimized.evaluate([3.0, 2.0])[0] == pytest.approx((((3 - 2)/4)**3 + 5)*2 - 1)
    assert optimized.evaluate([2.0, 3.0])[0] == pytest.approx((((2 - 3)/5)**3 + 5)*2 - 1)


def test_cse_preserves_signed_zero_in_constants_and_operand_order():
    x = Parameter(1.0).variable()
    program = compile_graph((x + 0.0, x + -0.0, 1.0 - x, x - 1.0))
    values = program.evaluate([-0.0])
    assert not np.signbit(values[0])
    assert np.signbit(values[1])
    np.testing.assert_array_equal(values[2:], [1.0, -1.0])


def test_optimizer_does_not_apply_algebraic_identities_to_nonfinite_runtime_values():
    x = Parameter(1.0).variable()
    program = compile_graph((x * 0, x - x, x / x))
    with np.errstate(invalid="ignore"):
        actual = program.evaluate([np.inf])
    assert all(np.isnan(value) for value in actual)


def test_array_constants_with_ambiguous_equality_are_not_merged_or_packed():
    first, second = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    x = Parameter(1.0).variable()
    program = compile_graph((Symbol("a") + x, Symbol("b") + x), {"a": first, "b": second})
    result = program.evaluate([2.0])
    np.testing.assert_allclose(result[0], [3.0, 4.0])
    np.testing.assert_allclose(result[1], [5.0, 6.0])
    np.testing.assert_array_equal(first, [1.0, 2.0])
    np.testing.assert_array_equal(second, [3.0, 4.0])


def test_optimized_and_unoptimized_complex_paths_have_independent_analytic_derivatives():
    x = Parameter(0.5).variable()
    shared = x + Symbol("z")
    roots = (shared**2, shared**2, shared * 2)
    raw, optimized = precompile(roots), compile_graph(roots)
    def cost(point, program):
        roots = program.evaluate(point[:1], {"z": point[1] + 1j*point[2]})
        return xp.real(roots[0] * xp.conj(roots[0]))
    def expected(point):
        return ((point[0] + point[1])**2 + point[2]**2)**2
    point = xp.array([0.5, 0.3, 0.2])
    for program in (raw, optimized):
        function = lambda point: cost(point, program)
        for transform in (lambda f: f, grad, hessian):
            reference = transform(expected)(point)
            np.testing.assert_allclose(transform(function)(point), reference, atol=1e-12)
            np.testing.assert_allclose(jit(transform(function))(point), reference, atol=1e-12)
