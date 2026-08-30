import pytest

from diffractix.composites import CompositeElement, Slab
from diffractix.elements import Interface, OpticalElement, Space, ThinLens
from diffractix.graph import Parameter
from diffractix.core.system import Placement, SourceInfo, System, SystemPlacement


class NestedComposite(CompositeElement):

    def __init__(self):
        self.lens = ThinLens(f=0.1, label="Lens")
        self.slab = Slab(d=0.01, n=1.5, label="Slab")
        super().__init__()


# ------------------
# RESOLVE ELEMENTS
# ------------------

def test_resolve_elements_preserves_optical_element():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens)

    original = system.placements[0]
    resolved = system._resolve_elements()

    assert resolved == (original,)
    assert resolved[0] is original
    assert resolved[0].element is lens


def test_resolve_elements_returns_tuple():
    system = System()
    system.add(ThinLens(f=0.1))

    resolved = system._resolve_elements()

    assert isinstance(resolved, tuple)


def test_resolve_elements_flattens_composite():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab)

    resolved = system._resolve_elements()

    assert tuple(placement.element for placement in resolved) == (
        slab.front,
        slab.body,
        slab.back,
    )


def test_resolve_elements_preserves_leaf_order():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    lens = ThinLens(f=0.1)

    system.add(lens)
    system.add(slab)

    resolved = system._resolve_elements()

    assert tuple(placement.element for placement in resolved) == (
        lens,
        slab.front,
        slab.body,
        slab.back,
    )


def test_resolve_elements_assigns_absolute_position_only_to_first_leaf():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab, z=0.5)

    resolved = system._resolve_elements()

    assert resolved[0].z == pytest.approx(0.5)
    assert resolved[1].z is None
    assert resolved[2].z is None


def test_resolve_elements_preserves_absolute_position_node():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    z = Parameter(0.5).variable()

    system.add(slab, z=z)

    resolved = system._resolve_elements()

    assert resolved[0].z is z


def test_resolve_elements_preserves_source_info():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab)

    source_info = system.placements[0].source_info
    resolved = system._resolve_elements()

    assert all(placement.source_info is source_info for placement in resolved)


def test_resolve_elements_records_composite_paths():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab)

    resolved = system._resolve_elements()

    assert tuple(placement.path for placement in resolved) == (
        "front",
        "body",
        "back",
    )


def test_resolve_elements_records_nested_composite_paths():
    system = System()
    composite = NestedComposite()
    system.add(composite)

    resolved = system._resolve_elements()

    assert tuple(placement.path for placement in resolved) == (
        "lens",
        "slab.front",
        "slab.body",
        "slab.back",
    )


def test_resolve_elements_does_not_modify_source_placement():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab, z=0.5)

    original = system.placements[0]
    system._resolve_elements()

    assert system.placements == (original,)
    assert original.element is slab
    assert original.z == pytest.approx(0.5)


def test_resolve_elements_does_not_modify_composite():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    original_elements = slab.elements

    system.add(slab)
    system._resolve_elements()

    assert slab.elements == original_elements


# --------------
# RESOLVE LAYOUT
# --------------

def test_resolve_layout_preserves_relative_element():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens)

    placements = system._resolve_elements()
    resolved = system._resolve_layout(placements)

    assert len(resolved) == 1
    assert resolved[0].element is lens
    assert resolved[0].z is None


def test_resolve_layout_returns_tuple():
    system = System()
    system.add(ThinLens(f=0.1))

    resolved = system._resolve_layout(system._resolve_elements())

    assert isinstance(resolved, tuple)


def test_resolve_layout_inserts_space_for_absolute_position():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())

    assert len(resolved) == 2
    assert isinstance(resolved[0].element, Space)
    assert resolved[1].element is lens
    assert resolved[0].element.d.value == pytest.approx(0.5)


def test_resolve_layout_generated_space_has_expected_label():
    system = System()
    lens = ThinLens(f=0.1, label="Lens")
    system.add(lens, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())

    assert resolved[0].element.label == "AutoSpace_to_Lens"


def test_resolve_layout_accounts_for_upstream_length():
    system = System()
    space = Space(d=0.2)
    lens = ThinLens(f=0.1)

    system.add(space)
    system.add(lens, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())

    assert len(resolved) == 3
    assert resolved[0].element is space
    assert isinstance(resolved[1].element, Space)
    assert resolved[2].element is lens
    assert resolved[1].element.d.value == pytest.approx(0.3)


def test_resolve_layout_accounts_for_multiple_absolute_positions():
    system = System()
    first_space = Space(d=0.2)
    lens1 = ThinLens(f=0.1)
    second_space = Space(d=0.1)
    lens2 = ThinLens(f=0.2)

    system.add(first_space)
    system.add(lens1, z=0.5)
    system.add(second_space)
    system.add(lens2, z=1.0)

    resolved = system._resolve_layout(system._resolve_elements())

    assert resolved[1].element.d.value == pytest.approx(0.3)
    assert resolved[4].element.d.value == pytest.approx(0.4)


def test_resolve_layout_preserves_variable_absolute_position_dependency():
    system = System()
    lens = ThinLens(f=0.1)
    z = Parameter(0.5).variable()

    system.add(lens, z=z)

    resolved = system._resolve_layout(system._resolve_elements())
    spacer = resolved[0].element

    assert spacer.d.value == pytest.approx(0.5)

    z.value = 0.8

    assert spacer.d.value == pytest.approx(0.8)


def test_resolve_layout_preserves_upstream_length_dependency():
    system = System()
    d = Parameter(0.2).variable()
    space = Space(d=d)
    lens = ThinLens(f=0.1)

    system.add(space)
    system.add(lens, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())
    spacer = resolved[1].element

    assert spacer.d.value == pytest.approx(0.3)

    d.value = 0.3

    assert spacer.d.value == pytest.approx(0.2)


def test_resolve_layout_supports_derived_absolute_position():
    system = System()
    anchor = Parameter(0.5)
    lens = ThinLens(f=0.1)

    system.add(lens, z=anchor + 0.2)

    resolved = system._resolve_layout(system._resolve_elements())

    assert resolved[0].element.d.value == pytest.approx(0.7)


def test_resolve_layout_removes_absolute_position_from_output():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())

    assert all(placement.z is None for placement in resolved)


def test_resolve_layout_preserves_source_info():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens, z=0.5)

    source_info = system.placements[0].source_info
    resolved = system._resolve_layout(system._resolve_elements())

    assert resolved[0].source_info is source_info
    assert resolved[1].source_info is source_info


def test_resolve_layout_preserves_path():
    system = System()
    slab = Slab(d=0.01, n=1.5)
    system.add(slab, z=0.5)

    resolved = system._resolve_layout(system._resolve_elements())

    assert resolved[0].path == "front"
    assert resolved[1].path == "front"
    assert resolved[2].path == "body"
    assert resolved[3].path == "back"


def test_resolve_layout_does_not_modify_input_placements():
    system = System()
    lens = ThinLens(f=0.1)
    system.add(lens, z=0.5)

    placements = system._resolve_elements()
    system._resolve_layout(placements)

    assert placements[0].z == pytest.approx(0.5)


# --------------------------
# RESOLVE REFRACTIVE INDICES
# --------------------------

def test_resolve_refractive_indices_returns_system_placements():
    system = System(ambient_n=1.0)
    lens = ThinLens(f=0.1)
    placement = Placement(element=lens)

    resolved = system._resolve_refractive_indices((placement,))

    assert len(resolved) == 1
    assert isinstance(resolved[0], SystemPlacement)
    assert resolved[0].placement is placement


def test_resolve_refractive_indices_inherits_ambient_medium():
    system = System(ambient_n=1.33)
    lens = ThinLens(f=0.1)
    placement = Placement(element=lens)

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].refractive_index is system.ambient_n
    assert resolved[0].refractive_index.value == pytest.approx(1.33)


def test_resolve_refractive_indices_space_inherits_ambient_medium():
    system = System(ambient_n=1.33)
    space = Space(d=0.1)
    placement = Placement(element=space)

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].refractive_index is system.ambient_n


def test_resolve_refractive_indices_does_not_bind_inherited_element():
    system = System(ambient_n=1.33)
    space = Space(d=0.1)

    assert space.n.node is None

    system._resolve_refractive_indices((Placement(element=space),))

    assert space.n.node is None


def test_resolve_refractive_indices_accepts_matching_explicit_medium():
    system = System(ambient_n=1.5)
    space = Space(d=0.1, n=1.5)
    placement = Placement(element=space)

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].refractive_index is space.n
    assert resolved[0].refractive_index.value == pytest.approx(1.5)


def test_resolve_refractive_indices_rejects_mismatched_explicit_medium():
    system = System(ambient_n=1.0)
    space = Space(d=0.1, n=1.5, label="Glass")
    placement = Placement(element=space)

    with pytest.raises(ValueError) as exc_info:
        system._resolve_refractive_indices((placement,))

    text = str(exc_info.value)

    assert "Refractive index mismatch" in text
    assert "Glass" in text
    assert "Upstream medium: n=1.0000" in text
    assert "Element requires: n=1.5000" in text


def test_resolve_refractive_indices_interface_changes_medium():
    system = System(ambient_n=1.0)
    interface = Interface(n1=1.0, n2=1.5)
    placement = Placement(element=interface)

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].refractive_index is interface.n2
    assert resolved[0].refractive_index.value == pytest.approx(1.5)


def test_resolve_refractive_indices_interface_updates_downstream_inheritance():
    system = System(ambient_n=1.0)
    interface = Interface(n1=1.0, n2=1.5)
    lens = ThinLens(f=0.1)

    placements = (
        Placement(element=interface),
        Placement(element=lens),
    )

    resolved = system._resolve_refractive_indices(placements)

    assert resolved[0].refractive_index is interface.n2
    assert resolved[1].refractive_index is interface.n2


def test_resolve_refractive_indices_handles_medium_chain():
    system = System(ambient_n=1.0)
    front = Interface(n1=1.0, n2=1.5)
    body = Space(d=0.1, n=1.5)
    back = Interface(n1=1.5, n2=1.0)
    lens = ThinLens(f=0.1)

    placements = (
        Placement(element=front),
        Placement(element=body),
        Placement(element=back),
        Placement(element=lens),
    )

    resolved = system._resolve_refractive_indices(placements)

    assert resolved[0].refractive_index.value == pytest.approx(1.5)
    assert resolved[1].refractive_index.value == pytest.approx(1.5)
    assert resolved[2].refractive_index.value == pytest.approx(1.0)
    assert resolved[3].refractive_index.value == pytest.approx(1.0)


def test_resolve_refractive_indices_preserves_variable_ambient_parameter():
    system = System(ambient_n=1.0, ambient_n_variable=True)
    lens = ThinLens(f=0.1)
    placement = Placement(element=lens)

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].refractive_index is system.ambient_n
    assert resolved[0].refractive_index.is_variable


def test_resolve_refractive_indices_preserves_explicit_index_graph():
    system = System(ambient_n=1.5)
    n = Parameter(1.5).variable()
    space = Space(d=0.1, n=n)

    resolved = system._resolve_refractive_indices((
        Placement(element=space),
    ))

    assert resolved[0].refractive_index is space.n
    assert resolved[0].refractive_index.node is n


def test_resolve_refractive_indices_preserves_placement_identity():
    system = System(ambient_n=1.0)
    lens = ThinLens(f=0.1)
    source_info = SourceInfo(file="example.py", line=10, call_index=0)
    placement = Placement(
        element=lens,
        source_info=source_info,
        path="lens",
    )

    resolved = system._resolve_refractive_indices((placement,))

    assert resolved[0].placement is placement
    assert resolved[0].placement.source_info is source_info
    assert resolved[0].placement.path == "lens"


def test_resolve_refractive_indices_preserves_element_order():
    system = System(ambient_n=1.0)
    first = ThinLens(f=0.1)
    second = ThinLens(f=0.2)
    third = ThinLens(f=0.3)

    placements = (
        Placement(element=first),
        Placement(element=second),
        Placement(element=third),
    )

    resolved = system._resolve_refractive_indices(placements)

    assert tuple(item.element for item in resolved) == (
        first,
        second,
        third,
    )


def test_resolve_refractive_indices_interface_requires_matching_input_medium():
    system = System(ambient_n=1.0)
    interface = Interface(n1=1.5, n2=2.0, label="Wrong Interface")
    placement = Placement(element=interface)

    with pytest.raises(ValueError):
        system._resolve_refractive_indices((placement,))