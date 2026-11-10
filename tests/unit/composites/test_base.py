"""Composite hierarchy, live parameters and recursive binding contracts."""

from dataclasses import dataclass, field
import pytest

from diffractix.composites.composite import CompositeElement
from diffractix.elements import Space
from diffractix.elements.element import OpticalElement
from diffractix.graph import Node, Parameter, InputNode


@dataclass(eq=False, kw_only=True)
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
    composite = NestedComposite(x=2.0)

    expected = (
        ("relay.input", composite.input),
        ("relay.inner.first", composite.inner.first),
        ("relay.inner.second", composite.inner.second),
        ("relay.output", composite.output),
    )
    actual = tuple(composite.walk("relay"))
    assert tuple(path for path, _ in actual) == tuple(path for path, _ in expected)
    assert all(leaf is expected_leaf for (_, leaf), (_, expected_leaf) in zip(actual, expected))


def test_nested_parameter_links_remain_live():
    composite = NestedComposite(x=2.0)

    composite.x = 4.0

    assert composite.input.x.value == 4.0
    assert composite.inner.first.x.value == 4.0
    assert composite.inner.second.x.value == 8.0
    assert composite.output.x.value == 4.0


# --------------------------
# EMPTY AND SHARED HIERARCHIES
# --------------------------

def test_empty_composite_has_no_children_or_symbolic_dependencies():
    composite = CompositeElement()

    assert composite.elements == composite.element_names == composite.named_elements == ()
    assert composite.leaf_elements == ()
    assert list(composite.walk()) == []
    assert len(composite) == 0
    assert composite.element_length == 0
    with pytest.raises(IndexError):
        composite[0]


class Group(CompositeElement):
    def __init__(self, **children):
        for name, child in children.items():
            setattr(self, name, child)
        super().__init__()


@dataclass(eq=False, kw_only=True)
class BoundLeaf(DummyLeaf):
    supplied_bindings: dict = field(default_factory=dict)

    @property
    def bindings(self):
        return self.supplied_bindings


def test_child_discovery_ignores_private_class_attributes_and_metadata():
    class WithMetadata(CompositeElement):
        class_child = DummyLeaf(x=9.0)

        def __init__(self):
            self.label = "Container"
            self._private = DummyLeaf(x=2.0)
            self.note = "ordinary data"
            self.public = DummyLeaf(x=1.0)
            super().__init__()

    composite = WithMetadata()
    assert composite.element_names == ("public",)
    assert len(composite.elements) == 1
    assert composite.elements[0] is composite.public


def test_repeated_child_is_a_distinct_path_and_length_occurrence():
    shared = Space(d=Parameter(0.02))
    composite = Group(first=shared, second=Group(repeated=shared))

    assert tuple(path for path, _ in composite.walk()) == ("first", "second.repeated")
    assert all(leaf is shared for leaf in composite.leaf_elements)
    assert composite.element_length.evaluate() == pytest.approx(0.04)
    shared.d = 0.03
    assert composite.element_length.evaluate() == pytest.approx(0.06)


def test_nested_lengths_retain_dependencies_without_double_counting_composites():
    distance = Parameter(0.02).variable()
    composite = Group(
        entrance=Space(d=distance),
        inner=Group(first=Space(d=2 * distance), second=Space(d=0.01)),
    )
    length = composite.element_length

    assert length.evaluate() == pytest.approx(0.07)
    assert length.has_variable_parameters()
    distance.value = 0.03
    assert length.evaluate() == pytest.approx(0.10)


def test_composite_constraints_are_local_ordered_and_instance_owned():
    first, second = DummyComposite(x=2.0), DummyComposite(x=2.0)
    lower, upper = first.x >= 0, first.x <= 3
    child_constraint = first.first.x >= 1
    first.first.require(child_constraint)

    assert first.require(lower).require(upper) is first
    assert len(first.constraints) == 2
    assert first.constraints[0] is lower and first.constraints[1] is upper
    assert second.constraints == ()
    assert len(first.first.constraints) == 1
    assert first.first.constraints[0] is child_constraint


def test_recursive_bindings_preserve_nodes_and_do_not_mutate_child_mappings():
    parameter = Parameter(0.2).variable()
    left_bindings, right_bindings = {"distance": parameter}, {("calibration", 1): 1.5}
    left = BoundLeaf(x=1.0, supplied_bindings=left_bindings)
    right = BoundLeaf(x=2.0, supplied_bindings=right_bindings)
    composite = Group(entrance=left, nested=Group(exit=right))

    bindings = composite.bindings

    assert set(bindings) == {"distance", ("calibration", 1)}
    assert bindings["distance"] is parameter
    assert bindings[("calibration", 1)] == 1.5
    with pytest.raises(TypeError):
        bindings["distance"] = 99.0
    assert composite.bindings is bindings   # gathered once, read-only
    assert left_bindings["distance"] is parameter
    assert tuple(left_bindings) == ("distance",)
    assert right_bindings == {("calibration", 1): 1.5}
    assert composite.bindings["distance"] is parameter


def test_direct_child_alias_contributes_bindings_only_once():
    child = BoundLeaf(x=1.0, supplied_bindings={"calibration": 1.5})
    composite = Group(first=child, repeated=child)

    assert composite.bindings == {"calibration": 1.5}
    assert len(composite.leaf_elements) == 2


@pytest.mark.parametrize("second_value", [1.5, 2.0])
def test_distinct_binding_providers_reject_duplicate_keys_even_if_values_match(second_value):
    first = BoundLeaf(x=1.0, supplied_bindings={"calibration": 1.5})
    second = BoundLeaf(x=2.0, supplied_bindings={"calibration": second_value})
    composite = Group(first=first, nested=Group(second=second))

    with pytest.raises(ValueError, match="Duplicate symbol binding.*calibration"):
        _ = composite.bindings

    assert first.bindings == {"calibration": 1.5}
    assert second.bindings == {"calibration": second_value}



def test_provider_shared_between_composite_branches_contributes_once():
    shared = BoundLeaf(x=1.0, supplied_bindings={"calibration": 1.5})
    left = Group(a=DummyLeaf(x=1.0), shared=shared)
    right = Group(b=DummyLeaf(x=2.0), shared=shared)
    assert Group(left=left, right=right).bindings == {"calibration": 1.5}
    assert left.bindings == right.bindings == {"calibration": 1.5}


def test_a_distinct_provider_with_the_same_key_still_conflicts_across_branches():
    left = Group(shared=BoundLeaf(x=1.0, supplied_bindings={"calibration": 1.5}))
    right = Group(other=BoundLeaf(x=2.0, supplied_bindings={"calibration": 1.5}))
    with pytest.raises(ValueError, match="Duplicate symbol binding.*calibration"):
        _ = Group(left=left, right=right).bindings
