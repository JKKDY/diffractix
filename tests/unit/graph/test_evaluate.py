from types import SimpleNamespace

from diffractix.backend import xp as anp
import numpy as np
import pytest
from diffractix.backend import grad, hessian, jit

from diffractix.graph import (
    InputNode,
    Literal,
    Parameter,
    Symbol,
    evaluate_ast,
)
from diffractix.graph.utils import ASTCycleError, UnresolvedInputError


def snapshot_info(value, parameter_index):
    """Create structural parameter metadata without importing Simulation types."""
    return SimpleNamespace(
        value=value,
        parameter_index=parameter_index,
    )


def test_evaluate_ast_literal_and_basic_arithmetic():
    expression = -(Literal(2.0) + 3.0) * 4.0

    assert evaluate_ast(expression, np.array([])) == -20.0


def test_evaluate_ast_fixed_parameter_uses_snapshot_value():
    parameter = Parameter(2.0, name="p")
    snapshot = {id(parameter): snapshot_info(2.0, None)}
    parameter.value = 10.0

    assert evaluate_ast(parameter * 3.0, np.array([]), snapshot) == 6.0


def test_evaluate_ast_variable_parameter_reads_canonical_theta_index():
    parameter = Parameter(1.0, name="p").variable()
    snapshot = {id(parameter): snapshot_info(1.0, 1)}

    assert evaluate_ast(
        parameter * 2.0,
        np.array([4.0, 3.0]),
        snapshot,
    ) == 6.0


def test_evaluate_ast_uses_live_fixed_parameter_without_snapshot_entry():
    parameter = Parameter(3.0, name="p")

    assert evaluate_ast(parameter * 2.0, np.array([])) == 6.0


def test_evaluate_ast_rejects_live_variable_without_snapshot_entry():
    parameter = Parameter(3.0, name="p").variable()

    with pytest.raises(ValueError, match="theta index cannot be determined"):
        evaluate_ast(parameter, np.array([]))


def test_evaluate_ast_reads_symbol_from_runtime_bindings():
    key = ("at", 42, None, "w")
    symbol = Symbol(key)

    assert evaluate_ast(
        symbol * 2.0,
        np.array([]),
        bindings={key: 1.5},
    ) == 3.0


@pytest.mark.parametrize("symbol", [Symbol("beam_width"), Symbol("beam_width", default=2.0)],
                         ids=["unbound", "default-is-metadata"])
def test_evaluate_ast_reports_missing_symbol_binding(symbol):
    with pytest.raises(
        KeyError,
        match="Missing runtime binding for Symbol key 'beam_width'",
    ):
        evaluate_ast(symbol, np.array([]))


def test_evaluate_ast_follows_input_node():
    handle = InputNode(Literal(2.0))

    assert evaluate_ast(handle * 3.0, np.array([])) == 6.0


def test_evaluate_ast_rejects_empty_input_node():
    with pytest.raises(UnresolvedInputError):
        evaluate_ast(InputNode(None), np.array([]))


def test_evaluate_ast_detects_cycles():
    handle = InputNode(None)
    handle.node = handle + 1.0

    with pytest.raises(ASTCycleError):
        evaluate_ast(handle, np.array([]))


def test_evaluate_ast_remains_autograd_differentiable():
    parameter = Parameter(1.0, name="p").variable()
    snapshot = {id(parameter): snapshot_info(1.0, 0)}
    expression = parameter**2 + 3.0 * parameter

    derivative = grad(lambda theta: evaluate_ast(expression, theta, snapshot))

    np.testing.assert_allclose(derivative(anp.array([4.0])), [11.0])


def test_context_alias_resolves_snapshot_parameters_before_runtime_bindings():
    variable = Parameter(1.0).variable()
    fixed = Parameter(2.0)
    snapshot = {
        id(variable): snapshot_info(1.0, 1),
        id(fixed): snapshot_info(2.0, None),
    }
    key = ("source", "position")
    context = {key: Symbol("expression"), "expression": InputNode(variable**2 + fixed)}
    variable.fixed()
    variable.value = 99.0
    fixed.value = 99.0
    point = anp.array([100.0, 3.0])

    def evaluate(theta):
        return evaluate_ast(
            Symbol(key) + Symbol("offset"), theta, snapshot,
            bindings={key: -99.0, "offset": 0.5}, context=context,
        )

    for function in (evaluate, jit(evaluate)):
        assert function(point) == pytest.approx(11.5)
    for function in (grad(evaluate), jit(grad(evaluate))):
        np.testing.assert_allclose(function(point), [0.0, 6.0])
    for function in (hessian(evaluate), jit(hessian(evaluate))):
        np.testing.assert_allclose(function(point), [[0.0, 0.0], [0.0, 2.0]])
    # Resolution is read-only, including the numerical runtime bindings.
    assert context["expression"].node.right is fixed


@pytest.mark.parametrize("cycle", ["self", "aliases", "expression"])
def test_evaluate_ast_detects_context_symbol_cycles(cycle):
    context = {
        "self": {"a": Symbol("a")},
        "aliases": {"a": Symbol("b"), "b": Symbol("a")},
        "expression": {"a": Symbol("a") + 1},
    }[cycle]
    with pytest.raises(ASTCycleError, match="Cycle"):
        evaluate_ast(Symbol("a"), anp.array([]), context=context)


def test_context_alias_can_resolve_a_runtime_array_without_coercion():
    values = anp.array([1.0, 2.0])
    result = evaluate_ast(
        Symbol("alias"), anp.array([]), context={"alias": Symbol("runtime")},
        bindings={"runtime": values},
    )
    assert result is values


def test_missing_symbol_in_context_expression_keeps_useful_error():
    with pytest.raises(KeyError, match="Missing runtime binding.*'missing'"):
        evaluate_ast(
            Symbol("alias"), anp.array([]), context={"alias": Symbol("missing") + 1},
        )
