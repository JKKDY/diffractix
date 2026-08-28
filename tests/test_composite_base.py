from dataclasses import dataclass

from diffractix.composites.composite import CompositeElement
from diffractix.elements.element import OpticalElement
from diffractix.graph import Node, Parameter, InputNode


@dataclass(kw_only=True)
class DummyLeaf(OpticalElement):

    x: Node

    @property
    def matrix(self):
        return (
            (1.0, self.x),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return 0.0


class DummyComposite(CompositeElement):

    x: Node

    def __init__(self, x):
        self.x = x
        self.label = "dummy"

        self.first = DummyLeaf(x=self.x)
        self.second = DummyLeaf(x=2 * self.x)

        super().__init__()


class NestedComposite(CompositeElement):

    x: Node

    def __init__(self, x):
        self.x = x

        self.input = DummyLeaf(x=self.x)
        self.inner = DummyComposite(x=self.x)
        self.output = DummyLeaf(x=self.x)

        super().__init__()


# ---------------------
# PARAMETER DECLARATION
# ---------------------

def test_node_annotation_declares_composite_parameter():
    composite = DummyComposite(x=2.0)

    assert composite.parameter_names == ("x",)
    assert isinstance(composite.x, InputNode)
    assert isinstance(composite.x.node, Parameter)
    assert composite.x.value == 2.0


def test_composite_parameter_is_owned_by_composite():
    composite = DummyComposite(x=2.0)

    assert composite.x.node.owner is composite
    assert composite.x.node.name == "x"


def test_ordinary_attributes_are_not_parameters():
    composite = DummyComposite(x=2.0)

    assert composite.label == "dummy"
    assert not isinstance(composite.label, InputNode)


def test_inherited_parameter_annotations_are_discovered():

    class ParentComposite(CompositeElement):

        x: Node

    class ChildComposite(ParentComposite):

        y: Node

        def __init__(self, x, y):
            self.x = x
            self.y = y
            super().__init__()

    composite = ChildComposite(x=1.0, y=2.0)

    assert composite.parameter_names == ("x", "y")


# ---------------
# CHILD DISCOVERY
# ---------------

def test_public_elements_are_discovered_as_children():
    composite = DummyComposite(x=2.0)

    assert composite.elements == (
        composite.first,
        composite.second,
    )


def test_element_names_follow_attribute_assignment_order():
    composite = DummyComposite(x=2.0)

    assert composite.element_names == (
        "first",
        "second",
    )


def test_named_elements_pairs_names_and_elements():
    composite = DummyComposite(x=2.0)

    assert composite.named_elements == (
        ("first", composite.first),
        ("second", composite.second),
    )


def test_non_element_attributes_are_not_discovered():
    composite = DummyComposite(x=2.0)

    assert "x" not in composite.element_names
    assert "label" not in composite.element_names


def test_private_element_attributes_are_not_discovered():

    class PrivateComposite(CompositeElement):

        def __init__(self):
            self.public = DummyLeaf(x=1.0)
            self._private = DummyLeaf(x=2.0)
            super().__init__()

    composite = PrivateComposite()

    assert composite.element_names == ("public",)
    assert composite.elements == (composite.public,)


# ------------------
# SEQUENCE INTERFACE
# ------------------

def test_len_reports_number_of_direct_children():
    composite = DummyComposite(x=2.0)

    assert len(composite) == 2


def test_iteration_returns_direct_children():
    composite = DummyComposite(x=2.0)

    assert list(composite) == [
        composite.first,
        composite.second,
    ]


def test_indexing_returns_direct_children():
    composite = DummyComposite(x=2.0)

    assert composite[0] is composite.first
    assert composite[1] is composite.second
    assert composite[-1] is composite.second


def test_slicing_returns_child_tuple_slice():
    composite = DummyComposite(x=2.0)

    assert composite[:] == (
        composite.first,
        composite.second,
    )


# -----------------
# PARAMETER LINKING
# -----------------

def test_child_parameter_can_reference_composite_parameter():
    composite = DummyComposite(x=2.0)

    assert composite.first.x.node is composite.x


def test_child_expression_can_depend_on_composite_parameter():
    composite = DummyComposite(x=2.0)

    assert composite.second.x.value == 4.0


def test_reassigning_composite_parameter_preserves_handle():
    composite = DummyComposite(x=2.0)
    handle = composite.x

    composite.x = 3.0

    assert composite.x is handle
    assert composite.x.value == 3.0


def test_reassigning_composite_parameter_updates_direct_child_link():
    composite = DummyComposite(x=2.0)

    composite.x = 3.0

    assert composite.first.x.value == 3.0


def test_reassigning_composite_parameter_updates_derived_child_expression():
    composite = DummyComposite(x=2.0)

    composite.x = 3.0

    assert composite.second.x.value == 6.0


# -----------------
# NESTED COMPOSITES
# -----------------

def test_nested_composite_remains_direct_child():
    composite = NestedComposite(x=2.0)

    assert composite.elements == (
        composite.input,
        composite.inner,
        composite.output,
    )

    assert composite.element_names == (
        "input",
        "inner",
        "output",
    )


def test_iteration_does_not_flatten_nested_composites():
    composite = NestedComposite(x=2.0)

    assert list(composite) == [
        composite.input,
        composite.inner,
        composite.output,
    ]


def test_walk_recursively_yields_leaf_elements():
    composite = NestedComposite(x=2.0)

    assert list(composite.walk()) == [
        ("input", composite.input),
        ("inner.first", composite.inner.first),
        ("inner.second", composite.inner.second),
        ("output", composite.output),
    ]


def test_walk_supports_prefix():
    composite = DummyComposite(x=2.0)

    assert list(composite.walk("relay")) == [
        ("relay.first", composite.first),
        ("relay.second", composite.second),
    ]


def test_leaf_elements_recursively_flattens_composite():
    composite = NestedComposite(x=2.0)

    assert composite.leaf_elements == (
        composite.input,
        composite.inner.first,
        composite.inner.second,
        composite.output,
    )


def test_nested_parameter_links_remain_live():
    composite = NestedComposite(x=2.0)

    composite.x = 4.0

    assert composite.input.x.value == 4.0
    assert composite.inner.first.x.value == 4.0
    assert composite.inner.second.x.value == 8.0
    assert composite.output.x.value == 4.0


# -------
# DISPLAY
# -------

def test_repr_contains_composite_type_and_length():
    composite = DummyComposite(x=2.0)

    text = repr(composite)

    assert "DummyComposite" in text
    assert "len=2" in text