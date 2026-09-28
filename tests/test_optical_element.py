import pytest

from dataclasses import dataclass

from diffractix.elements.element import OpticalElement, parameter
from diffractix.graph import Node, Parameter, InputNode
from diffractix.graph.utils import (
    ASTCycleError,
    UnresolvedInputError,
    UnsupportedNodeError,
)


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


@dataclass(kw_only=True)
class ValidationElement(OpticalElement):
    matrix_value: object = ((1.0, 0.0), (0.0, 1.0))
    length_value: object = 0.0
    refractive_index_value: object | None = None

    @property
    def matrix(self):
        return self.matrix_value

    @property
    def element_length(self):
        return self.length_value

    @property
    def element_refractive_index(self):
        return self.refractive_index_value


class UnsupportedTestNode(Node):
    pass


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
    requirement_a = element.x >= 0.0
    requirement_b = element.x <= 2.0

    result = element.require(requirement_a, requirement_b)

    assert result is element
    assert element.requirements[0] is requirement_a
    assert element.requirements[1] is requirement_b


def test_requirements_are_not_shared_between_elements():
    first = DummyElement(x=1.0)
    second = DummyElement(x=1.0)
    requirement = first.x >= 0.0

    first.require(requirement)

    assert first.requirements[0] is requirement
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


# ----------------------
# BUILD REPRESENTATION
# ----------------------

def test_builtin_element_graph_validation_defaults_to_enabled():
    assert OpticalElement.validate_graph_inputs is True
    assert DummyElement.validate_graph_inputs is True


def test_graph_validation_can_be_disabled_per_subclass():
    @dataclass(kw_only=True)
    class UnvalidatedElement(ValidationElement, validate_graph_inputs=False):
        pass

    invalid = UnvalidatedElement(
        matrix_value=((True,),),
        length_value="invalid",
        refractive_index_value=False,
    )

    assert not UnvalidatedElement.validate_graph_inputs
    assert DummyElement.validate_graph_inputs
    invalid._validate_for_build()


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    (
        (None, TypeError, "matrix must be a 2x2"),
        (((1.0, 0.0),), ValueError, "exactly 2 rows"),
        (((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), ValueError, "exactly 2 rows"),
        (((1.0,), (0.0, 1.0)), ValueError, "matrix row 0"),
        (((1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), ValueError, "matrix row 0"),
        ((1.0, (0.0, 1.0)), TypeError, "matrix row 0"),
        (((True, 0.0), (0.0, 1.0)), TypeError, r"matrix\[0\]\[0\]"),
        ((("bad", 0.0), (0.0, 1.0)), TypeError, r"matrix\[0\]\[0\]"),
    ),
)
def test_build_validation_rejects_invalid_matrix_structure(matrix, error, message):
    element = ValidationElement(matrix_value=matrix)

    with pytest.raises(error, match=message):
        element._validate_for_build()


@pytest.mark.parametrize("length", (True, 1j, "0.1"))
def test_build_validation_rejects_invalid_element_length(length):
    element = ValidationElement(length_value=length)

    with pytest.raises(TypeError, match="element_length"):
        element._validate_for_build()


@pytest.mark.parametrize("index", (True, 1j, "1.5"))
def test_build_validation_rejects_invalid_refractive_index(index):
    element = ValidationElement(refractive_index_value=index)

    with pytest.raises(TypeError, match="element_refractive_index"):
        element._validate_for_build()


def test_build_validation_accepts_numeric_scalars_nodes_and_optional_index():
    element = ValidationElement(
        matrix_value=((1, Parameter(0.1)), (0.0, 1.0)),
        length_value=Parameter(0.0),
        refractive_index_value=None,
    )

    element._validate_for_build()


def test_build_validation_rejects_empty_input_nodes():
    element = ValidationElement(matrix_value=((InputNode(None), 0.0), (0.0, 1.0)))

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        element._validate_for_build()


def test_build_validation_allows_empty_optional_refractive_index_handle():
    element = ValidationElement(refractive_index_value=InputNode(None))

    element._validate_for_build()


def test_build_validation_rejects_graph_cycles():
    cyclic = InputNode(None)
    cyclic.node = cyclic
    element = ValidationElement(matrix_value=((cyclic, 0.0), (0.0, 1.0)))

    with pytest.raises(ASTCycleError, match="Cycle detected"):
        element._validate_for_build()


def test_build_validation_rejects_unsupported_node_types():
    element = ValidationElement(
        matrix_value=((UnsupportedTestNode(), 0.0), (0.0, 1.0))
    )

    with pytest.raises(UnsupportedNodeError, match="Unsupported AST node type"):
        element._validate_for_build()


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
