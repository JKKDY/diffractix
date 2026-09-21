from types import SimpleNamespace

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad

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


def test_evaluate_ast_reports_missing_symbol_binding():
    symbol = Symbol("beam_width")

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
