import pytest

from dataclasses import dataclass

from diffractix.elements.element import OpticalElement, parameter
from diffractix.graph import Node, Parameter, InputNode
from diffractix.graph import (
    ASTCycleError,
    UnresolvedInputError,
    UnsupportedNodeError,
)


@dataclass(eq=False, kw_only=True)
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


@dataclass(eq=False, kw_only=True)
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


@dataclass(eq=False, kw_only=True)
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
    def refractive_index(self):
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

    @dataclass(eq=False, kw_only=True)
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


# ------------------
# ELEMENT PROPERTIES
# ------------------

def test_refractive_index_defaults_to_none():
    element = DummyElement(x=1.0)

    assert element.refractive_index is None


def test_optical_element_requires_matrix():

    @dataclass(eq=False, kw_only=True)
    class MissingMatrixElement(OpticalElement):

        x: Node

        @property
        def element_length(self):
            return 0.0

    with pytest.raises(TypeError):
        MissingMatrixElement(x=1.0)


def test_optical_element_requires_element_length():

    @dataclass(eq=False, kw_only=True)
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
        element.validate()


@pytest.mark.parametrize("length", (True, 1j, "0.1"))
def test_build_validation_rejects_invalid_element_length(length):
    element = ValidationElement(length_value=length)

    with pytest.raises(TypeError, match="element_length"):
        element.validate()


@pytest.mark.parametrize("index", (True, 1j, "1.5"))
def test_build_validation_rejects_invalid_refractive_index(index):
    element = ValidationElement(refractive_index_value=index)

    with pytest.raises(TypeError, match="refractive_index"):
        element.validate()


def test_build_validation_accepts_numeric_scalars_nodes_and_optional_index():
    element = ValidationElement(
        matrix_value=((1, Parameter(0.1)), (0.0, 1.0)),
        length_value=Parameter(0.0),
        refractive_index_value=None,
    )

    element.validate()


def test_build_validation_rejects_empty_input_nodes():
    element = ValidationElement(matrix_value=((InputNode(None), 0.0), (0.0, 1.0)))

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        element.validate()


def test_build_validation_allows_empty_optional_refractive_index_handle():
    element = ValidationElement(refractive_index_value=InputNode(None))

    element.validate()


def test_build_validation_rejects_graph_cycles():
    cyclic = InputNode(None)
    cyclic.node = cyclic
    element = ValidationElement(matrix_value=((cyclic, 0.0), (0.0, 1.0)))

    with pytest.raises(ASTCycleError, match="Cycle detected"):
        element.validate()


def test_build_validation_rejects_unsupported_node_types():
    element = ValidationElement(
        matrix_value=((UnsupportedTestNode(), 0.0), (0.0, 1.0))
    )

    with pytest.raises(UnsupportedNodeError, match="Unsupported AST node type"):
        element.validate()


@pytest.mark.parametrize("dependency", ["length", "index"])
def test_build_validation_walks_dependencies_outside_the_matrix(dependency):
    empty = InputNode(None)
    kwargs = {"length_value": empty} if dependency == "length" else {"refractive_index_value": empty + 1}
    element = ValidationElement(**kwargs)

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        element.validate()


def test_optional_index_does_not_hide_the_same_empty_handle_in_required_matrix():
    empty = InputNode(None)
    element = ValidationElement(
        matrix_value=((1.0, empty), (0.0, 1.0)),
        refractive_index_value=empty,
    )

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        element.validate()


def test_build_validation_accepts_shared_graph_dependencies_and_complex_coefficients():
    parameter = Parameter(0.1)
    shared = parameter + 1
    element = ValidationElement(
        matrix_value=((shared, 1j * shared), (shared, shared)),
        length_value=shared,
        refractive_index_value=shared,
    )

    element.validate()

    assert shared.left is parameter
    assert parameter.value == 0.1


def test_build_validation_rejects_array_as_a_single_matrix_coefficient():
    import numpy as np

    element = ValidationElement(matrix_value=((1.0, np.array([0.0])), (0.0, 1.0)))
    with pytest.raises(TypeError, match=r"matrix\[0\]\[1\]"):
        element.validate()


# -------
# DISPLAY
# -------

@pytest.mark.parametrize("kind, value, marker", [
    ("fixed", 2.0, "[FIX]"),
    ("variable", 2.0, "[VAR]"),
    ("expression", 4.0, "[EXPR]"),
])
def test_string_representation_describes_element_and_parameter(kind, value, marker):
    parameter = 2 * Parameter(2.0) if kind == "expression" else 2.0
    element = DummyElement(x=parameter, label="Example")
    if kind == "variable":
        element.variable("x")

    text = str(element)

    assert "DummyElement" in text
    assert "Example" in text
    assert f"L={value:g}" in text
    assert f"x={value:g}" in text
    assert marker in text
