"""Array constants retain shape, dtype and derivatives throughout compilation."""

import numpy as np
import pytest

from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.graph import (
    Constant, Literal, Parameter, Symbol, compile_graph, fuse, optimize,
    precompile, remap_parameters, remap_runtime_symbols,
)


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
@pytest.mark.parametrize("value", [
    np.array([1, 2], dtype=np.int16),
    np.array([1, 2], dtype=np.uint32),
    np.array([[1, 2], [3, 4]], dtype=np.float32),
    np.array([1 + 2j, 3 - 4j], dtype=np.complex64),
    np.empty((0, 2), dtype=np.float64),
])
def test_direct_array_outputs_preserve_shape_dtype_and_immutable_storage(compiler, value):
    literal = Literal(value)
    program = compiler((literal, Literal(2), literal))
    value[...] = 99
    actual = program.evaluate([])
    assert isinstance(actual, tuple)
    assert actual[0].shape == actual[2].shape == literal.value.shape
    assert actual[0].dtype == actual[2].dtype == literal.value.dtype
    np.testing.assert_array_equal(actual[0], literal.value)
    np.testing.assert_array_equal(actual[2], literal.value)
    assert actual[1] == 2
    assert not actual[0].flags.writeable


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_compile_time_array_binding_is_an_independent_snapshot(compiler):
    original = np.arange(6, dtype=np.float32)
    binding = original[::2]
    bindings = {"array": binding}
    program = compiler((Symbol("array"), Symbol("array") + 1), bindings)
    original[:] = 100
    bindings["array"] = np.array([-1])
    values = program.evaluate([])
    np.testing.assert_array_equal(values[0], [0, 2, 4])
    np.testing.assert_array_equal(values[1], [1, 3, 5])
    assert values[0].dtype == np.dtype("float32")
    assert not values[0].flags.writeable
    assert not np.shares_memory(values[0], binding)
    assert binding.flags.writeable


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_optimization_preserves_broadcasting_and_backend_dtype_promotion(dtype):
    first = np.array([[1], [2]], dtype=dtype)
    second = np.array([3, 4, 5], dtype=np.float32)
    roots = (-(Literal(first) + second) / 2, Literal(first) * second)
    raw = precompile(roots)
    program = optimize(raw)
    expected = (-(xp.array(first) + xp.array(second)) / 2, xp.array(first) * xp.array(second))
    for compiled in (raw, program):
        for actual, reference in zip(compiled.evaluate([]), expected):
            np.testing.assert_allclose(actual, reference)
            assert actual.shape == (2, 3)
            assert actual.dtype == reference.dtype


def test_cse_handles_different_arrays_and_preserves_signed_zero():
    variable = Parameter(1).variable()
    roots = (
        variable * np.array([0.0, -0.0]),
        variable * np.array([-0.0, 0.0]),
        variable + np.array([1, 2]),
        variable + np.array([3, 4]),
    )
    values = compile_graph(roots).evaluate([1.0])
    np.testing.assert_array_equal(np.signbit(values[0]), [False, True])
    np.testing.assert_array_equal(np.signbit(values[1]), [True, False])
    np.testing.assert_array_equal(values[2], [2, 3])
    np.testing.assert_array_equal(values[3], [4, 5])


def test_register_reuse_preserves_earlier_array_outputs_and_shared_subexpressions():
    variable = Parameter(1).variable()
    first = variable + np.array([1., 2.])
    square = first * first
    roots = (first, square / (variable + 2), square, first, square - first)
    raw = precompile(roots)
    program = optimize(raw)
    assert program._register_count < raw._register_count
    for compiled in (raw, program):
        expected = (np.array([3., 4.]), np.array([9., 16.]) / 4,
                    np.array([9., 16.]), np.array([3., 4.]), np.array([6., 12.]))
        for actual, reference in zip(compiled.evaluate([2.]), expected):
            np.testing.assert_allclose(actual, reference)


def test_array_constants_survive_parameter_symbol_remapping_and_program_fusion():
    first, second = Parameter(1).variable(), Parameter(2).variable()
    one = precompile((first + np.array([1., 2.]) + Symbol("runtime"),))
    two = precompile((second * np.array([3., 4.]),))
    mapping = {id(second): 0, id(first): 1}
    one = remap_runtime_symbols(remap_parameters(one, mapping), {"runtime": 0})
    two = remap_runtime_symbols(remap_parameters(two, mapping), {"runtime": 0})
    program = optimize(fuse(one, two))
    values = program.evaluate([5., 7.], {"runtime": xp.array([10., 20.])})
    np.testing.assert_allclose(values[0], [18, 29])
    np.testing.assert_allclose(values[1], [15, 20])


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_array_literal_paths_have_analytical_derivatives_and_support_jit(compiler):
    variable = Parameter(0.5).variable()
    coefficients = np.array([1., 2., -3.])
    offsets = np.array([0.5, -1., 2.])
    program = compiler((variable * coefficients + offsets,))

    def residual(theta):
        return program.evaluate(theta)[0]

    def cost(theta):
        return xp.sum(residual(theta) ** 2)

    point = xp.array([0.5])
    np.testing.assert_allclose(jit(residual)(point), [1., 0., 0.5])
    np.testing.assert_allclose(jit(jacobian(residual))(point), coefficients[:, None])
    np.testing.assert_allclose(jit(grad(cost))(point), [-1.])
    np.testing.assert_allclose(jit(hessian(cost))(point), [[28.]])
    np.testing.assert_allclose(jit(grad(lambda theta: xp.sum(hessian(cost)(theta))))(point), [0.])


def test_complex_array_literals_preserve_real_cost_derivatives():
    variable = Parameter(0.5).variable()
    coefficients = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    program = compile_graph((variable * coefficients,))

    def cost(theta):
        value = program.evaluate(theta)[0]
        return xp.sum(xp.real(value * xp.conj(value)))

    point = xp.array([0.5])
    assert jit(cost)(point) == pytest.approx(7.5)
    np.testing.assert_allclose(jit(grad(cost))(point), [30.])
    np.testing.assert_allclose(jit(hessian(cost))(point), [[60.]])


def test_compiling_inside_jit_does_not_capture_active_tracers_as_constants():
    variable = Parameter(1).variable()
    root = (Literal(np.array([1., 2.])) + np.array([3., 4.])) * variable

    def function(theta):
        return compile_graph((root,)).evaluate(theta)[0]

    np.testing.assert_allclose(jit(function)(xp.array([2.])), [8., 12.])
    np.testing.assert_allclose(jit(jacobian(function))(xp.array([2.])), [[4.], [6.]])


def test_program_constant_owns_input_array_without_changing_the_callers_flags():
    original = np.array([1., 2.])
    constant = Constant(original)
    original[0] = 99
    np.testing.assert_array_equal(constant.value, [1., 2.])
    assert original.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        constant.value[0] = 9
