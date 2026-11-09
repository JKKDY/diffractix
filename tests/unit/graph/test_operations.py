"""Every graph operation: values, gradients and Hessians against analytic formulas.

Each expression is evaluated through the interpreter, a raw precompiled Program
and an optimized Program, eagerly and under JIT. References are written
independently with NumPy, never by calling diffractix.
"""
import numpy as np
import pytest

from diffractix.backend import grad, hessian, jit, xp
from diffractix.graph import Parameter, compile_graph, precompile

# Autograd warns when a Hessian is identically zero (linear operations).
pytestmark = pytest.mark.filterwarnings("ignore:Output seems independent of input\\.:UserWarning")

X, Y = 0.7, 1.3


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# name: (build, point, value, gradient, hessian) with point a scalar x.
UNARY = {
    "neg": (lambda p: -p, X, -X, -1, 0),
    "pos": (lambda p: +p, X, X, 1, 0),
    "abs-positive": (abs, X, X, 1, 0),
    "abs-negative": (abs, -X, X, -1, 0),
    "sin": (lambda p: p.sin(), X, np.sin(X), np.cos(X), -np.sin(X)),
    "cos": (lambda p: p.cos(), X, np.cos(X), -np.sin(X), -np.cos(X)),
    "tan": (lambda p: p.tan(), X, np.tan(X), 1 / np.cos(X)**2, 2 * np.tan(X) / np.cos(X)**2),
    "sinh": (lambda p: p.sinh(), X, np.sinh(X), np.cosh(X), np.sinh(X)),
    "cosh": (lambda p: p.cosh(), X, np.cosh(X), np.sinh(X), np.cosh(X)),
    "tanh": (lambda p: p.tanh(), X, np.tanh(X), 1 - np.tanh(X)**2,
             -2 * np.tanh(X) * (1 - np.tanh(X)**2)),
    "exp": (lambda p: p.exp(), X, np.exp(X), np.exp(X), np.exp(X)),
    "log": (lambda p: p.log(), X, np.log(X), 1 / X, -1 / X**2),
    "sqrt": (lambda p: p.sqrt(), X, np.sqrt(X), 0.5 / np.sqrt(X), -0.25 * X**-1.5),
    "sigmoid": (lambda p: p.sigmoid(), X, _sigmoid(X), _sigmoid(X) * (1 - _sigmoid(X)),
                _sigmoid(X) * (1 - _sigmoid(X)) * (1 - 2 * _sigmoid(X))),
}

# name: (build, value, gradient [d/dx, d/dy], hessian) at (X, Y); max/min avoid ties.
BINARY = {
    "add": (lambda a, b: a + b, X + Y, [1, 1], [[0, 0], [0, 0]]),
    "sub": (lambda a, b: a - b, X - Y, [1, -1], [[0, 0], [0, 0]]),
    "mul": (lambda a, b: a * b, X * Y, [Y, X], [[0, 1], [1, 0]]),
    "div": (lambda a, b: a / b, X / Y, [1 / Y, -X / Y**2],
            [[0, -1 / Y**2], [-1 / Y**2, 2 * X / Y**3]]),
    "pow": (lambda a, b: a ** b, X**Y, [Y * X**(Y - 1), X**Y * np.log(X)],
            [[Y * (Y - 1) * X**(Y - 2), X**(Y - 1) * (1 + Y * np.log(X))],
             [X**(Y - 1) * (1 + Y * np.log(X)), X**Y * np.log(X)**2]]),
    "maximum": (lambda a, b: a.maximum(b), Y, [0, 1], [[0, 0], [0, 0]]),
    "minimum": (lambda a, b: a.minimum(b), X, [1, 0], [[0, 0], [0, 0]]),
}


def _evaluators(expression, parameters):
    """Scalar functions of a theta vector ordered like `parameters`."""
    def interpreter(theta):
        return expression.evaluate(
            parameter_values={id(p): theta[i] for i, p in enumerate(parameters)})

    def program_function(program):
        order = [next(i for i, p in enumerate(parameters) if id(p) == key)
                 for key, _ in sorted(program.parameter_indices.items(), key=lambda item: item[1])]
        return lambda theta: program.evaluate(xp.stack([theta[i] for i in order]))[0]

    return {
        "interpreter": interpreter,
        "precompile": program_function(precompile((expression,))),
        "compile_graph": program_function(compile_graph((expression,))),
    }


def _check(expression, parameters, point, value, gradient, hessian_matrix):
    point = xp.array(point, dtype=float)
    for path, function in _evaluators(expression, parameters).items():
        for mode, wrap in (("eager", lambda f: f), ("jit", jit)):
            label = f"{path}/{mode}"
            np.testing.assert_allclose(wrap(function)(point), value, rtol=1e-12, err_msg=label)
            np.testing.assert_allclose(wrap(grad(function))(point), gradient,
                                       rtol=1e-11, atol=1e-14, err_msg=label)
            np.testing.assert_allclose(wrap(hessian(function))(point), hessian_matrix,
                                       rtol=1e-10, atol=1e-13, err_msg=label)


@pytest.mark.parametrize("name", list(UNARY))
def test_unary_operation_values_and_derivatives(name):
    build, point, value, first, second = UNARY[name]
    parameter = Parameter(point).variable()
    _check(build(parameter), [parameter], [point], value, [first], [[second]])


@pytest.mark.parametrize("name", list(BINARY))
def test_binary_operation_values_and_derivatives(name):
    build, value, gradient, hessian_matrix = BINARY[name]
    left, right = Parameter(X).variable(), Parameter(Y).variable()
    _check(build(left, right), [left, right], [X, Y], value, gradient, hessian_matrix)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "pow"])
def test_binary_operations_with_a_constant_on_either_side(name):
    build, _, _, _ = BINARY[name]
    parameter = Parameter(X).variable()
    for expression, reference in ((build(parameter, Y), build(X, Y)), (build(Y, parameter), build(Y, X))):
        for function in _evaluators(expression, [parameter]).values():
            np.testing.assert_allclose(function(xp.array([X])), reference, rtol=1e-12)
