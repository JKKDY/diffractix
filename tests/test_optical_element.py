import pytest

from dataclasses import dataclass

from diffractix.elements.element import OpticalElement, parameter
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
class ExplicitParameterElement(OpticalElement):

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


# ---------------------
# PARAMETER DECLARATION
# ---------------------

def test_node_annotation_declares_parameter():
    element = DummyElement(x=2.0)

    assert element.parameter_names == ("x",)
    assert isinstance(element.x, InputNode)
    assert isinstance(element.x.node, Parameter)
    assert element.x.value == 2.0


def test_parameter_helper_declares_parameter():
    element = ExplicitParameterElement()

    assert element.parameter_names == ("gain",)
    assert isinstance(element.gain, InputNode)
    assert isinstance(element.gain.node, Parameter)
    assert element.gain.value == 2.0


def test_ordinary_fields_remain_ordinary_python_values():
    element = DummyElement(x=1.0, metadata="hello")

    assert element.metadata == "hello"
    assert not isinstance(element.metadata, InputNode)


def test_node_union_annotation_declares_parameter():

    @dataclass(kw_only=True)
    class OptionalParameterElement(OpticalElement):

        x: Node | None = None

        @property
        def matrix(self):
            return ((1.0, 0.0), (0.0, 1.0))

        @property
        def element_length(self):
            return 0.0

    element = OptionalParameterElement()

    assert element.parameter_names == ("x",)
    assert isinstance(element.x, InputNode)
    assert element.x.node is None


# ------
# LABELS
# ------

def test_automatic_labels_are_unique():
    first = DummyElement(x=1.0)
    second = DummyElement(x=1.0)

    assert first.label.startswith("DummyElement")
    assert second.label.startswith("DummyElement")
    assert first.label != second.label


def test_explicit_label_is_preserved():
    element = DummyElement(x=1.0, label="Test Element")

    assert element.label == "Test Element"


# ------------
# REQUIREMENTS
# ------------

def test_requirements_are_empty_by_default():
    element = DummyElement(x=1.0)

    assert element.requirements == ()


def test_require_adds_persistent_requirements():
    element = DummyElement(x=1.0)
    requirement_a = object()
    requirement_b = object()

    result = element.require(requirement_a, requirement_b)

    assert result is element
    assert element.requirements == (requirement_a, requirement_b)


def test_requirements_are_not_shared_between_elements():
    first = DummyElement(x=1.0)
    second = DummyElement(x=1.0)
    requirement = object()

    first.require(requirement)

    assert first.requirements == (requirement,)
    assert second.requirements == ()


# ------------------
# ELEMENT PROPERTIES
# ------------------

def test_matrix_is_declarative():
    element = DummyElement(x=2.0)

    assert element.matrix == (
        (1.0, element.x),
        (0.0, 1.0),
    )


def test_element_length_is_declarative():
    element = DummyElement(x=2.0)

    assert element.element_length is element.x


def test_element_refractive_index_defaults_to_none():
    element = DummyElement(x=1.0)

    assert element.element_refractive_index is None


def test_optical_element_requires_matrix():

    @dataclass(kw_only=True)
    class MissingMatrixElement(OpticalElement):

        x: Node

        @property
        def element_length(self):
            return 0.0

    with pytest.raises(TypeError):
        MissingMatrixElement(x=1.0)


def test_optical_element_requires_element_length():

    @dataclass(kw_only=True)
    class MissingLengthElement(OpticalElement):

        x: Node

        @property
        def matrix(self):
            return ((1.0, 0.0), (0.0, 1.0))

    with pytest.raises(TypeError):
        MissingLengthElement(x=1.0)


# ----------
# VALIDATION
# ----------

def test_graph_validation_is_enabled_by_default():
    assert DummyElement.validate_graph_inputs is True


def test_graph_validation_can_be_disabled_per_class():

    @dataclass(kw_only=True)
    class CustomElement(OpticalElement, validate_graph_inputs=False):

        x: Node

        @property
        def matrix(self):
            return ((1.0, self.x), (0.0, 1.0))

        @property
        def element_length(self):
            return 0.0

    assert CustomElement.validate_graph_inputs is False


def test_graph_validation_setting_does_not_modify_parent():
    @dataclass(kw_only=True)
    class CustomElement(OpticalElement, validate_graph_inputs=False):

        x: Node

        @property
        def matrix(self):
            return ((1.0, self.x), (0.0, 1.0))

        @property
        def element_length(self):
            return 0.0

    assert CustomElement.validate_graph_inputs is False
    assert DummyElement.validate_graph_inputs is True


# -------
# DISPLAY
# -------

def test_string_representation_contains_element_type_and_label():
    element = DummyElement(x=2.0, label="Example")

    text = str(element)

    assert "DummyElement" in text
    assert "Example" in text


def test_string_representation_contains_length():
    element = DummyElement(x=2.0, label="Example")

    assert "L=2" in str(element)


def test_string_representation_contains_fixed_parameter():
    element = DummyElement(x=2.0, label="Example")

    text = str(element)

    assert "x=2" in text
    assert "[FIX]" in text


def test_string_representation_contains_variable_parameter():
    element = DummyElement(x=2.0, label="Example")
    element.variable("x")

    text = str(element)

    assert "x=2" in text
    assert "[VAR]" in text


def test_string_representation_marks_expression_parameter():
    x = Parameter(2.0, name="x")
    element = DummyElement(x=2 * x, label="Example")

    text = str(element)

    assert "x=4" in text
    assert "[EXPR]" in text