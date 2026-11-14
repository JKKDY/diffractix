"""Direct graph interpretation: live values, explicit bindings and pure evaluation."""

import numpy as np
import pytest

from diffractix.backend import grad, hessian, jit, xp
from diffractix.graph import (
    ASTCycleError, InputNode, Literal, Node, Parameter, Symbol,
    UnresolvedInputError, UnsupportedNodeError, evaluate_ast,
)


def test_literal_arithmetic_and_live_fixed_or_variable_parameters():
    fixed = Parameter(2.0)
    variable = Parameter(3.0).variable()
    root = -(fixed + Literal(1.0)) * variable**2
    assert evaluate_ast(root) == -27.0
    fixed.value = 4.0
    variable.fixed()
    variable.value = 2.0
    assert root.evaluate() == -20.0


def test_explicit_parameter_values_override_every_parameter_without_mutation():
    fixed, variable = Parameter(2.0), Parameter(3.0).variable()
    root = fixed + variable**2
    values = {id(fixed): 10.0, id(variable): 4.0}
    assert root.evaluate(parameter_values=values) == 26.0
    assert (fixed.value, variable.value, fixed.is_variable, variable.is_variable) == (2, 3, False, True)
    assert values == {id(fixed): 10.0, id(variable): 4.0}


@pytest.mark.parametrize("variable", [False, True])
def test_explicit_parameter_mapping_requires_fixed_and_variable_entries(variable):
    parameter = Parameter(2.0, name="missing", variable=variable)
    with pytest.raises(KeyError, match="Missing value for Parameter.*missing"):
        parameter.evaluate(parameter_values={})


def test_bindings_follow_nodes_and_aliases():
    parameter = Parameter(2.0)
    key = ("input", 42, "offset")
    expression = 3 * Symbol(key) + 1
    bindings = {key: Symbol("inner"), "inner": InputNode(parameter + 1)}
    assert expression.evaluate(bindings=bindings) == 10.0
    parameter.value = 4.0
    assert expression.evaluate(bindings=bindings) == 16.0
    assert bindings[key].key == "inner"


def test_node_valued_bindings_are_evaluated_with_parameter_overrides():
    parameter = Parameter(2.0)
    symbol = Symbol("derived")
    bindings = {"derived": parameter * 3}
    assert symbol.evaluate(bindings=bindings, parameter_values={id(parameter): 4.0}) == 12.0
    assert symbol.evaluate(bindings={"derived": 0.0}, parameter_values={}) == 0.0


def test_runtime_array_binding_is_not_packed_or_cast():
    values = xp.array([1.0, 2.0], dtype="float32")
    symbol = Symbol("array")
    assert symbol.evaluate(bindings={"array": values}) is values
    result = (2 * symbol + 1).evaluate(bindings={"array": values})
    np.testing.assert_allclose(result, [3.0, 5.0])
    assert result.dtype == values.dtype


def test_missing_symbol_binding_identifies_key():
    with pytest.raises(KeyError, match="required"):
        Symbol("required").evaluate()


def test_input_handle_swap_is_seen_on_next_evaluation():
    handle = InputNode(Parameter(2.0))
    root = handle * 3
    assert root.evaluate() == 6.0
    handle.node = Literal(4.0)
    assert root.evaluate() == 12.0
    handle.node = None
    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        root.evaluate()


@pytest.mark.parametrize("kind", ["handle", "alias", "self"])
def test_cycle_detection_is_by_active_identity_and_recovery_is_possible(kind):
    if kind == "handle":
        root = InputNode(None)
        root.node = root + 1
        bindings = {}
    elif kind == "alias":
        root = Symbol("a")
        bindings = {"a": Symbol("b"), "b": root + 1}
    else:
        root = Symbol("a")
        bindings = {"a": root + 1}
    with pytest.raises(ASTCycleError, match="Cycle"):
        root.evaluate(bindings=bindings)
    if kind == "handle":
        root.node = Literal(3)
        assert root.evaluate() == 3
    else:
        assert root.evaluate(bindings={"a": 3}) == 3


def test_shared_dag_is_not_a_cycle_and_values_are_not_cached_between_calls():
    parameter = Parameter(2.0)
    shared = parameter + 1
    root = shared * shared + shared
    assert root.evaluate() == 12.0
    parameter.value = 3.0
    assert root.evaluate() == 20.0


def test_unsupported_node_fails_clearly():
    class Unknown(Node):
        pass
    with pytest.raises(UnsupportedNodeError, match="Unknown"):
        Unknown().evaluate()


def test_interpreter_preserves_gradients_hessians_nested_derivatives_and_jit():
    parameter = Parameter(0.0).variable()
    expression = parameter**3 + Symbol("coefficient") * parameter**2
    def value(point):
        return expression.evaluate(
            bindings={"coefficient": point[1]}, parameter_values={id(parameter): point[0]},
        )
    point = xp.array([2.0, 3.0])
    for function in (value, jit(value)):
        assert function(point) == pytest.approx(20.0)
    for function in (grad(value), jit(grad(value))):
        np.testing.assert_allclose(function(point), [24.0, 4.0])
    for function in (hessian(value), jit(hessian(value))):
        np.testing.assert_allclose(function(point), [[18.0, 4.0], [4.0, 0.0]])
    nested = grad(lambda t: grad(lambda s: value(xp.stack((s, t[1]))))(t[0]))
    np.testing.assert_allclose(nested(point), [18.0, 4.0])
