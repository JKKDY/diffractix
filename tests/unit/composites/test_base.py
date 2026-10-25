from dataclasses import dataclass
import pytest

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
    assert composite.x.node.owner is composite
    assert composite.x.node.name == "x"
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

def test_child_views_and_sequence_access_preserve_declaration_order():
    composite = DummyComposite(x=2.0)
    children = (composite.first, composite.second)
    assert composite.elements == children
    assert composite.element_names == ("first", "second")
    assert composite.named_elements == (
        ("first", composite.first),
        ("second", composite.second),
    )
    assert len(composite) == 2
    assert tuple(composite) == children
    assert composite[0] is composite.first
    assert composite[1] is composite[-1] is composite.second
    assert composite[:] == children
    assert composite[1:] == (composite.second,)
    with pytest.raises(IndexError):
        composite[2]


def test_private_element_attributes_are_not_discovered():

    class PrivateComposite(CompositeElement):

        def __init__(self):
            self.public = DummyLeaf(x=1.0)
            self._private = DummyLeaf(x=2.0)
            super().__init__()

    composite = PrivateComposite()

    assert composite.element_names == ("public",)
    assert composite.elements == (composite.public,)


# -----------------
# PARAMETER LINKING
# -----------------

def test_reassignment_keeps_direct_and_derived_child_links_live():
    composite = DummyComposite(x=2.0)
    assert composite.first.x.node is composite.x
    assert composite.second.x.value == 4.0
    handle = composite.x

    composite.x = 3.0

    assert composite.x is handle
    assert composite.x.value == 3.0
    assert composite.first.x.node is handle
    assert composite.first.x.value == 3.0
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
    assert list(composite) == [
        composite.input,
        composite.inner,
        composite.output,
    ]


def test_walk_recursively_yields_leaf_elements():
    composite = NestedComposite(x=2.0)

    expected = [
        ("input", composite.input),
        ("inner.first", composite.inner.first),
        ("inner.second", composite.inner.second),
        ("output", composite.output),
    ]
    assert list(composite.walk()) == expected
    assert composite.leaf_elements == tuple(element for _, element in expected)


def test_walk_supports_prefix():
    composite = DummyComposite(x=2.0)

    assert list(composite.walk("relay")) == [
        ("relay.first", composite.first),
        ("relay.second", composite.second),
    ]


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
