import pytest

from dataclasses import dataclass

from diffractix.elements.base import OpticalElement, parameter
from diffractix.graph import Node, Parameter, InputNode


@dataclass(kw_only=True)
class DummyElement(OpticalElement):
    x: Node
    metadata: str = "test"

    @property
    def matrix(self):
        return (
            (1.0, self.x),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return self.x


@dataclass(kw_only=True)
class ExplicitInputElement(OpticalElement):
    gain: float = parameter(2.0)
    metadata: str = "test"

    @property
    def matrix(self):
        return (
            (self.gain, 0.0),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return 0.0


# -----------------------
# INPUT DECLARATION
# -----------------------

def test_node_annotation_declares_graph_input():
    """Node-annotated fields should become stable graph inputs."""
    element = DummyElement(x=2.0)

    assert element.input_names == ("x",)
    assert isinstance(element.x, InputNode)
    assert isinstance(element.x.node, Parameter)
    assert element.x.value == 2.0


def test_parameter_helper_declares_graph_input():
    """parameter(...) should explicitly declare a graph input."""
    element = ExplicitInputElement()

    assert element.input_names == ("gain",)
    assert isinstance(element.gain, InputNode)
    assert isinstance(element.gain.node, Parameter)
    assert element.gain.value == 2.0


def test_ordinary_fields_remain_ordinary_python_values():
    """Fields that are not graph inputs should not be wrapped."""
    element = DummyElement(x=1.0, metadata="hello")

    assert element.metadata == "hello"
    assert not isinstance(element.metadata, InputNode)


# -----------------------
# INPUT TARGETS
# -----------------------

def test_scalar_input_creates_element_owned_parameter():
    """Raw scalar inputs should create Parameters local to the element."""
    element = DummyElement(x=3.0)

    assert isinstance(element.x.node, Parameter)
    assert element.x.node.owner is element
    assert element.x.node.name == "x"
    assert not element.x.node.is_variable


def test_existing_parameter_preserves_identity_and_owner():
    """Externally created Parameters should be referenced rather than adopted."""
    parameter = Parameter(3.0, name="shared")
    element = DummyElement(x=parameter)

    assert element.x.node is parameter
    assert parameter.owner is None


def test_expression_input_preserves_expression_identity():
    """Element inputs may directly reference derived AST expressions."""
    x = Parameter(2.0, name="x")
    expression = 3 * x

    element = DummyElement(x=expression)

    assert element.x.node is expression


def test_none_creates_empty_input_handle():
    """None should represent an unresolved graph-input socket."""

    @dataclass(kw_only=True)
    class OptionalInputElement(OpticalElement):
        x: Node | None = None

        @property
        def matrix(self):
            return ((1.0, 0.0), (0.0, 1.0))

        @property
        def element_length(self):
            return 0.0

    element = OptionalInputElement()

    assert isinstance(element.x, InputNode)
    assert element.x.node is None


# -----------------------
# HANDLE STABILITY
# -----------------------

def test_scalar_reassignment_preserves_input_handle():
    """Reassignment should replace the target without replacing the handle."""
    element = DummyElement(x=1.0)
    handle = element.x

    element.x = 2.0

    assert element.x is handle
    assert isinstance(element.x.node, Parameter)
    assert element.x.value == 2.0
    assert element.x.node.owner is element


def test_node_reassignment_preserves_input_handle():
    """Existing Nodes should be hot-swappable behind the same handle."""
    element = DummyElement(x=1.0)
    handle = element.x

    external = Parameter(4.0, name="external")
    element.x = external

    assert element.x is handle
    assert element.x.node is external


def test_existing_expression_tracks_reassigned_handle():
    """
    Expressions created from an element input should follow later changes to
    that input until the graph is compiled.
    """
    element = DummyElement(x=2.0)
    expression = 3 * element.x

    assert expression.value == 6.0

    element.x = 5.0

    assert expression.value == 15.0


# -----------------------
# VARIABILITY
# -----------------------

def test_inputs_are_fixed_by_default():
    element = DummyElement(x=1.0)

    assert not element.x.node.is_variable
    assert element.variable_input_names == ()


def test_variable_and_fixed_toggle_direct_parameter():
    element = DummyElement(x=1.0)

    assert element.variable() is element
    assert element.x.node.is_variable
    assert element.variable_input_names == ("x",)

    assert element.fixed() is element
    assert not element.x.node.is_variable
    assert element.variable_input_names == ()


def test_named_variable_selection():
    @dataclass(kw_only=True)
    class TwoInputElement(OpticalElement):
        x: Node
        y: Node

        @property
        def matrix(self):
            return ((self.x, 0.0), (0.0, self.y))

        @property
        def element_length(self):
            return 0.0

    element = TwoInputElement(x=1.0, y=2.0)

    element.variable("y")

    assert not element.x.node.is_variable
    assert element.y.node.is_variable
    assert element.variable_input_names == ("y",)


def test_named_expression_input_cannot_be_marked_variable():
    """
    A derived input is not itself an independent optimization Parameter.
    """
    x = Parameter(2.0, name="x")
    element = DummyElement(x=2 * x)

    with pytest.raises(TypeError):
        element.variable("x")


def test_unknown_input_name_is_rejected():
    element = DummyElement(x=1.0)

    with pytest.raises(ValueError):
        element.variable("missing")


# -----------------------
# INTROSPECTION
# -----------------------

def test_values_reports_directly_evaluable_inputs():
    element = DummyElement(x=2.5)

    assert element.values == (2.5,)


def test_automatic_labels_are_unique():
    first = DummyElement(x=1.0)
    second = DummyElement(x=1.0)

    assert first.label.startswith("DummyElement")
    assert second.label.startswith("DummyElement")
    assert first.label != second.label


def test_explicit_label_is_preserved():
    element = DummyElement(x=1.0, label="Test Element")

    assert element.label == "Test Element"


def test_string_representation_contains_useful_element_state():
    element = DummyElement(x=2.0, label="Example")

    text = str(element)

    assert "DummyElement" in text
    assert "Example" in text
    assert "x=2" in text
    assert "[FIX]" in text


def test_graph_validation_can_be_disabled_per_class():
    @dataclass(kw_only=True)
    class CustomElement(
        OpticalElement,
        validate_graph_inputs=False,
    ):
        x: Node

        @property
        def matrix(self):
            return ((1.0, self.x), (0.0, 1.0))

        @property
        def element_length(self):
            return 0.0

    assert CustomElement.validate_graph_inputs is False