"""Shared optical parameter ownership, stable handles, and discovery."""

from dataclasses import dataclass

import pytest

from diffractix.core import OpticalBase
from diffractix.graph import Node, Parameter, InputNode, Symbol


class DummyElement(OpticalBase):

    def __init__(self, x):
        self.x = x

    @classmethod
    def _get_parameter_names(cls):
        return ("x",)


class TwoParameterElement(OpticalBase):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def _get_parameter_names(cls):
        return ("x", "y")


# ----------------
# PARAMETER ACCESS
# ----------------

def test_parameters_returns_stable_handles():
    element = TwoParameterElement(x=1.0, y=2.0)

    assert element.parameter_names == ("x", "y")
    assert element.parameters == (element.x, element.y)
    assert all(isinstance(handle, InputNode) for handle in element.parameters)


# --------------------
# PARAMETER ASSIGNMENT
# --------------------

def test_scalar_creates_element_owned_parameter():
    element = DummyElement(x=3.0)

    assert isinstance(element.x, InputNode)
    assert isinstance(element.x.node, Parameter)
    assert element.x.node.owner is element
    assert element.x.node.name == "x"
    assert element.x.value == 3.0
    assert element.x.is_variable is False


def test_existing_parameter_preserves_identity_and_owner():
    parameter = Parameter(3.0, name="shared")
    element = DummyElement(x=parameter)

    assert element.x.node is parameter
    assert parameter.owner is None


def test_expression_preserves_identity():
    x = Parameter(2.0, name="x")
    expression = 3 * x
    element = DummyElement(x=expression)

    assert element.x.node is expression


def test_none_creates_empty_input_handle():
    element = DummyElement(x=None)

    assert isinstance(element.x, InputNode)
    assert element.x.node is None


@pytest.mark.parametrize("value", ["invalid", True])
def test_invalid_parameter_value_is_rejected(value):
    with pytest.raises(TypeError):
        DummyElement(x=value)


# ----------------
# HANDLE STABILITY
# ----------------

def test_scalar_reassignment_preserves_input_handle():
    element = DummyElement(x=1.0)
    handle = element.x

    element.x = 2.0

    assert element.x is handle
    assert isinstance(element.x.node, Parameter)
    assert element.x.value == 2.0
    assert element.x.node.owner is element


def test_node_reassignment_preserves_input_handle():
    element = DummyElement(x=1.0)
    handle = element.x
    external = Parameter(4.0, name="external")

    element.x = external

    assert element.x is handle
    assert element.x.node is external


def test_existing_expression_tracks_reassigned_handle():
    element = DummyElement(x=2.0)
    expression = 3 * element.x

    assert expression.value == 6.0

    element.x = 5.0

    assert expression.value == 15.0


# -----------
# VARIABILITY
# -----------

def test_variable_and_fixed_toggle_direct_parameter():
    element = DummyElement(x=1.0)

    assert element.variable() is element
    assert element.x.node.is_variable

    assert element.fixed() is element
    assert not element.x.node.is_variable


def test_named_variable_selection():
    element = TwoParameterElement(x=1.0, y=2.0)

    element.variable("y")

    assert not element.x.node.is_variable
    assert element.y.node.is_variable


def test_named_fixed_selection():
    element = TwoParameterElement(x=1.0, y=2.0)
    element.variable()

    element.fixed("x")

    assert not element.x.node.is_variable
    assert element.y.node.is_variable


@pytest.mark.parametrize("method", ["variable", "fixed"])
def test_named_expression_parameter_cannot_change_variability(method):
    x = Parameter(2.0, name="x")
    element = DummyElement(x=2 * x)

    with pytest.raises(TypeError):
        getattr(element, method)("x")


def test_unnamed_variable_skips_derived_parameters():
    x = Parameter(2.0, name="x")
    element = TwoParameterElement(x=2 * x, y=3.0)

    element.variable()

    assert not x.is_variable
    assert element.y.node.is_variable


def test_unknown_parameter_name_is_rejected():
    element = DummyElement(x=1.0)

    with pytest.raises(ValueError):
        element.variable("missing")


# -------------
# INTROSPECTION
# -------------

@pytest.mark.parametrize("value,expected", [(3.5, 3.5), (None, None), (Symbol("missing"), None)])
def test_values_reports_evaluable_values_and_unresolved_inputs(value, expected):
    element = TwoParameterElement(x=2.5, y=value)

    assert element.values == (2.5, expected)


# --------------------
# CLASS INITIALIZATION
# --------------------

def test_discovery_cache_is_independent_for_parent_child_and_sibling_classes():
    @dataclass(eq=False)
    class Parent(OpticalBase):
        x: Node = 1.0

    assert Parent().parameter_names == ("x",)

    @dataclass(eq=False)
    class Child(Parent):
        y: Node = 2.0

    @dataclass(eq=False)
    class Sibling(Parent):
        z: Node = 3.0

    assert Child().parameter_names == ("x", "y")
    assert Sibling().parameter_names == ("x", "z")
    assert Parent().parameter_names == ("x",)
