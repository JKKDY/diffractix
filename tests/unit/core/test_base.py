"""Shared optical parameter ownership, stable handles, and discovery."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest

from diffractix.core import OpticalBase, parameter
from diffractix.graph import Node, Parameter, InputNode, Symbol


class DummyElement(OpticalBase):
    x: Node

    def __init__(self, x):
        self.x = x


class TwoParameterElement(OpticalBase):
    x: Node
    y: Node

    def __init__(self, x, y):
        self.x = x
        self.y = y


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


def test_unnamed_variable_skips_derived_parameters():
    x = Parameter(2.0, name="x")
    element = TwoParameterElement(x=2 * x, y=3.0)

    element.variable()

    assert not x.is_variable
    assert element.y.node.is_variable


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


def test_discovery_combines_annotations_metadata_and_inherited_fields():
    @dataclass(eq=False)
    class Parent(OpticalBase):
        inherited: Node = 1.0
        ordinary: float = 2.0
        graph_constant: ClassVar[Node] = Parameter(7.0)

    @dataclass(eq=False)
    class Child(Parent):
        explicit: float = parameter(3.0)
        optional: Node | None = None
        specialized: Parameter = 4.0
        samples: np.ndarray = field(default_factory=lambda: np.array([1.0]))
        phase: complex = 1j
        enabled: bool = True

    child = Child()

    assert child.parameter_names == ("inherited", "explicit", "optional", "specialized")
    assert all(isinstance(handle, InputNode) for handle in child.parameters)
    assert child.values == (1.0, 3.0, None, 4.0)
    assert child.graph_constant is Parent.graph_constant
    assert child.ordinary == 2.0 and not isinstance(child.ordinary, Node)
    np.testing.assert_array_equal(child.samples, [1.0])
    assert child.phase == 1j and child.enabled is True


@pytest.mark.parametrize("invalid", [True, "invalid", 1j, np.array([1.0]), object()])
def test_failed_assignment_keeps_handle_target_and_downstream_expression(invalid):
    element = DummyElement(2.0)
    handle, target = element.x, element.x.node
    expression = 3 * handle

    with pytest.raises(TypeError, match="Parameter"):
        element.x = invalid

    assert element.x is handle
    assert handle.node is target
    assert expression.evaluate() == 6.0


def test_handle_can_be_cleared_and_repopulated_without_detaching_expressions():
    element = DummyElement(2.0)
    handle = element.x
    expression = handle + 1

    element.x = None
    assert element.x is handle and handle.node is None
    assert element.values == (None,)

    element.x = 5.0
    assert element.x is handle
    assert expression.evaluate() == 6.0


@pytest.mark.parametrize("method", ["variable", "fixed"])
def test_unknown_names_are_validated_before_changing_any_selected_parameter(method):
    element = TwoParameterElement(1.0, 2.0)
    if method == "fixed":
        element.variable()
    before = (element.x.is_variable, element.y.is_variable)

    with pytest.raises(ValueError, match="missing"):
        getattr(element, method)("x", "missing")

    assert (element.x.is_variable, element.y.is_variable) == before


@pytest.mark.parametrize("method", ["variable", "fixed"])
def test_bulk_variability_skips_empty_and_derived_fields_but_named_selection_rejects_them(method):
    external = Parameter(2.0).variable()
    element = TwoParameterElement(x=external * 2, y=None)

    assert getattr(element, method)() is element
    assert external.is_variable
    assert element.y.node is None
    for name in ("x", "y"):
        with pytest.raises(TypeError, match="directly contain"):
            getattr(element, method)(name)


def test_constraints_are_ordered_per_instance_tuple_views_without_copying_expressions():
    first, second = DummyElement(1.0), DummyElement(2.0)
    lower, upper = first.x >= 0, first.x <= 3

    assert first.require(lower) is first
    previous = first.constraints
    assert first.require(upper) is first

    assert isinstance(first.constraints, tuple)
    assert len(previous) == 1 and previous[0] is lower
    assert len(first.constraints) == 2
    assert first.constraints[0] is lower and first.constraints[1] is upper
    assert second.constraints == ()
    first.x = 4.0
    assert lower.left.evaluate() == 4.0


def test_ordinary_fields_remain_unwrapped_even_when_assigned_a_node():
    @dataclass(eq=False)
    class Metadata(OpticalBase):
        gain: Node = 2.0
        note: object = None

    element = Metadata()
    expression = Parameter(3.0) * 2
    element.note = expression

    assert element.parameter_names == ("gain",)
    assert element.note is expression
