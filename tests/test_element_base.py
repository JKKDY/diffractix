import pytest

from diffractix.elements.base import ElementBase
from diffractix.graph import Parameter, InputNode


class DummyElement(ElementBase):

    def __init__(self, x):
        self.x = x

    @classmethod
    def _get_parameter_names(cls):
        return ("x",)


class TwoParameterElement(ElementBase):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def _get_parameter_names(cls):
        return ("x", "y")


# ----------------
# PARAMETER ACCESS
# ----------------

def test_parameter_names():
    element = DummyElement(x=2.0)

    assert element.parameter_names == ("x",)


def test_parameters_returns_stable_handles():
    element = TwoParameterElement(x=1.0, y=2.0)

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


def test_invalid_parameter_value_is_rejected():
    with pytest.raises(TypeError):
        DummyElement(x="invalid")


def test_bool_parameter_value_is_rejected():
    with pytest.raises(TypeError):
        DummyElement(x=True)


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

def test_parameters_are_fixed_by_default():
    element = DummyElement(x=1.0)

    assert not element.x.node.is_variable


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


def test_named_expression_parameter_cannot_be_marked_variable():
    x = Parameter(2.0, name="x")
    element = DummyElement(x=2 * x)

    with pytest.raises(TypeError):
        element.variable("x")


def test_named_expression_parameter_cannot_be_marked_fixed():
    x = Parameter(2.0, name="x")
    element = DummyElement(x=2 * x)

    with pytest.raises(TypeError):
        element.fixed("x")


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

def test_values_reports_directly_evaluable_parameters():
    element = TwoParameterElement(x=2.5, y=3.5)

    assert element.values == (2.5, 3.5)


def test_values_reports_none_for_unresolved_parameters():
    element = TwoParameterElement(x=2.5, y=None)

    assert element.values == (2.5, None)


# --------------------
# CLASS INITIALIZATION
# --------------------

def test_subclass_resets_parameter_name_cache():
    DummyElement._parameter_names = ("cached",)

    class ChildElement(DummyElement):
        pass

    assert ChildElement.__dict__["_parameter_names"] is None