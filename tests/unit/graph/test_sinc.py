"""The unnormalized sinc must retain its removable-limit derivatives."""
from math import factorial

import numpy as np
import pytest

from diffractix.backend import grad, hessian, jit, xp
from diffractix.graph import Parameter, Symbol, compile_ast


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_sinc_values_across_small_argument_branch(dtype):
    graph = compile_ast([Symbol("x").sinc()])
    values = np.array([-1.0, -0.14, -0.02, -0.01, -1e-8, 0, 1e-8, 0.01, 0.02, 0.14, 1.0], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1e-4j
    expected = np.sinc(values / np.pi)
    evaluate = lambda x: graph.evaluate([], bindings={"x": x})[0]
    for function in (evaluate, jit(evaluate)):
        actual = function(xp.array(values))
        assert actual.dtype == values.dtype
        np.testing.assert_allclose(actual, expected, rtol=3 * np.finfo(dtype).eps, atol=3 * np.finfo(dtype).eps)


@pytest.mark.parametrize("point", [0.0, -1e-8, 1e-8, -0.01, 0.01, -0.02, 0.02])
def test_sinc_first_and_second_derivatives_match_taylor_series(point):
    p = Parameter(0.0).variable()
    graph = compile_ast([p.sinc()])
    evaluate = lambda x: graph.evaluate(xp.reshape(x, (1,)))[0]
    # Independent series through x**12 is accurate at every test point.
    coefficients = [(-1)**(k // 2) / factorial(k + 1) if k % 2 == 0 else 0 for k in range(13)]
    for order, derivative in ((1, grad(evaluate)), (2, hessian(evaluate))):
        expected = np.polynomial.polynomial.polyval(
            point, np.polynomial.polynomial.polyder(coefficients, m=order)
        )
        for function in (derivative, jit(derivative)):
            np.testing.assert_allclose(function(point), expected, rtol=1e-9, atol=1e-13)


def test_sinc_nested_fourth_derivative_at_zero():
    graph = compile_ast([Parameter(0.0).variable().sinc()])
    derivative = lambda x: graph.evaluate(xp.reshape(x, (1,)))[0]
    for _ in range(4):
        derivative = grad(derivative)
    for function in (derivative, jit(derivative)):
        np.testing.assert_allclose(function(0.0), 1 / 5, rtol=1e-12)


def test_sinc_large_argument_does_not_overflow_inactive_series():
    graph = compile_ast([Symbol("x").sinc()])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = graph.evaluate([], bindings={"x": 1e100})[0]
    np.testing.assert_allclose(actual, np.sin(1e100) / 1e100, rtol=1e-12, atol=0)
