"""Solver expression optimization versus the previous public-array path."""

from types import SimpleNamespace

import numpy as np
from diffractix.backend import xp
from diffractix.backend import grad, hessian, jacobian
import numpy as numpy
import pytest

from diffractix.graph import InputNode, Literal, Parameter, Symbol
from diffractix.solver.context import SolverContext
from diffractix.solver.solver import Solver
from tests.helpers.solver import (
    make_solver, make_solver_with_reordered_parameters, packed_expression_reference,
)


def assert_finite_match(actual, expected):
    assert numpy.all(numpy.isfinite(actual))
    assert numpy.all(numpy.isfinite(expected))
    numpy.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=False)


def test_scalar_expression_does_not_construct_local_or_output_arrays(monkeypatch):
    solver, _, parameter = make_solver()
    evaluate = solver._compile_node(parameter ** 2 + parameter)
    theta = xp.array([3.0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Scalar expressions must not construct temporary arrays")

    monkeypatch.setattr("diffractix.graph.compile.xp.array", forbidden)
    monkeypatch.setattr("diffractix.solver.solver.xp.stack", forbidden)
    assert evaluate(SolverContext(theta, solver.simulation.run)) == 12.0


def test_projection_aliases_snapshot_and_runtime_symbols_match_canonical_theta(monkeypatch):
    solver, lens, first, second = make_solver_with_reordered_parameters()
    fixed = Parameter(0.5)
    solver.simulation.parameter_info[id(fixed)] = SimpleNamespace(value=0.5, parameter_index=None)
    solver.simulation.compile_context = {"offset": fixed + Literal(0.3)}
    alias = InputNode(node=first)
    shared = 2 * alias + second
    function = lambda ctx: shared ** 3 + shared * Symbol("offset") + second * ctx.at(lens, occurrence=1).w
    optimized = solver._compile_callable(function)
    with monkeypatch.context() as reference:
        reference.setattr(Solver, "_compile_node", packed_expression_reference)
        packed = solver._compile_callable(function)

    # All declarative changes must remain irrelevant to the compiled snapshot.
    alias.node = second
    first.fixed()
    fixed.value = 99.0
    solver.simulation.compile_context["offset"] = Literal(99.0)
    theta = xp.array([0.7, 0.2])  # canonical [second, first], not AST discovery order
    fast = lambda t: optimized(SolverContext(t, solver.simulation.run))
    old = lambda t: packed(SolverContext(t, solver.simulation.run))
    expected = lambda t: (2 * t[1] + t[0]) ** 3 + 0.8 * (2 * t[1] + t[0]) + t[0] * (t[0] + 2)
    assert_finite_match(fast(theta), expected(theta))
    for differentiate in (grad, hessian, lambda f: jacobian(jacobian(hessian(f)))):
        assert_finite_match(differentiate(fast)(theta), differentiate(old)(theta))
        assert_finite_match(differentiate(fast)(theta), differentiate(expected)(theta))
    nested = lambda t, evaluate: xp.sum(grad(evaluate)(t) ** 2)
    assert_finite_match(hessian(lambda t: nested(t, fast))(theta), hessian(lambda t: nested(t, old))(theta))


@pytest.mark.parametrize("binding", [
    numpy.float32(0.25), numpy.float64(0.25), 2, True,
    numpy.complex64(0.25j), numpy.complex128(1 + 0.25j),
    numpy.array(0.25), numpy.array([0.25, 0.5], dtype="float32"),
    numpy.arange(12).reshape(3, 4)[:, ::-2],
])
def test_runtime_single_root_retains_dtype_shape_and_copy_behavior(binding, monkeypatch):
    solver, lens, _ = make_solver()
    solver.simulation.run = lambda theta: SimpleNamespace(at=lambda *args: SimpleNamespace(w=binding))
    function = lambda ctx: ctx.at(lens).w
    optimized = solver._compile_callable(function)
    with monkeypatch.context() as reference:
        reference.setattr(Solver, "_compile_node", packed_expression_reference)
        packed = solver._compile_callable(function)
    theta = xp.array([1.0])
    actual = optimized(SolverContext(theta, solver.simulation.run))
    expected = packed(SolverContext(theta, solver.simulation.run))
    numpy.testing.assert_array_equal(actual, expected)
    assert np.shape(actual) == np.shape(expected)
    assert numpy.asarray(actual).dtype == numpy.asarray(expected).dtype
    if isinstance(binding, numpy.ndarray):
        snapshot = numpy.array(expected, copy=True)
        binding[...] = 99
        numpy.testing.assert_array_equal(actual, snapshot)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "bool", "complex128"])
def test_tuple_projection_preserves_canonical_input_dtype(dtype):
    solver, _, parameter = make_solver()
    expression = parameter
    optimized = solver._compile_node(expression)
    packed = packed_expression_reference(solver, expression)
    theta = xp.array([1], dtype=dtype)
    actual = optimized(SolverContext(theta, solver.simulation.run))
    expected = packed(SolverContext(theta, solver.simulation.run))
    numpy.testing.assert_array_equal(actual, expected)
    assert numpy.asarray(actual).dtype == numpy.asarray(expected).dtype


def test_constant_expression_uses_empty_projection_without_running_simulation():
    solver, _, _ = make_solver()
    def forbidden(theta):
        raise AssertionError("Constant evaluation must not propagate")
    evaluate = solver._compile_node(Literal(7.0))
    result = evaluate(SolverContext(xp.array([2.0]), forbidden))
    assert result == 7.0
    assert numpy.asarray(result).dtype == numpy.dtype("float64")


def test_array_runtime_expression_and_nested_derivatives_match_packed_reference(monkeypatch):
    solver, lens, first, second = make_solver_with_reordered_parameters()
    solver.simulation.run = lambda t: SimpleNamespace(
        at=lambda *args: SimpleNamespace(w=xp.stack((t[0] ** 2, t[1] ** 3))))
    function = lambda ctx: (first + second) ** 2 * ctx.at(lens).w
    optimized = solver._compile_callable(function)
    with monkeypatch.context() as reference:
        reference.setattr(Solver, "_compile_node", packed_expression_reference)
        packed = solver._compile_callable(function)
    fast = lambda t: optimized(SolverContext(t, solver.simulation.run))
    old = lambda t: packed(SolverContext(t, solver.simulation.run))
    theta = xp.array([0.7, 0.2])
    assert_finite_match(fast(theta), old(theta))
    assert_finite_match(jacobian(fast)(theta), jacobian(old)(theta))
    cost = lambda t, f: xp.sum(f(t) ** 2)
    for differentiate in (grad, hessian, lambda f: jacobian(jacobian(hessian(f)))):
        assert_finite_match(differentiate(lambda t: cost(t, fast))(theta),
                            differentiate(lambda t: cost(t, old))(theta))


def test_single_root_missing_binding_and_invalid_shape_errors_remain_clear():
    from diffractix.graph import compile_ast

    graph = compile_ast((Symbol("runtime"),))
    with pytest.raises(KeyError, match="runtime"):
        graph._evaluate_sequence(())
    with pytest.raises(ValueError, match="Expected 0 variable values"):
        graph._evaluate_sequence((1.0,), bindings={"runtime": 2.0})
    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        graph._evaluate_sequence((), bindings={"runtime": [[1.0], [2.0, 3.0]]})
