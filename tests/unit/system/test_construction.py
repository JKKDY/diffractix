"""Declarative System registration, bindings, constraints and validation."""

from dataclasses import dataclass

import numpy as np
import pytest

from diffractix import GaussianBeam, Ray
from diffractix.composites import CompositeElement, Slab
from diffractix.elements import OpticalElement, Plane, Space, ThinLens
from diffractix.graph import Literal, Parameter, Symbol
from diffractix.system import AMBIENT_N, System, SystemValidationError
from diffractix.system.info import CallSite, Placement
from diffractix.system.symbols import make_symbol


@dataclass(eq=False, kw_only=True)
class FailingElement(OpticalElement):
    @property
    def matrix(self):
        return ((1.0, 0.0), (0.0, 1.0))

    @property
    def element_length(self):
        return 0.0

    def validate(self):
        raise ValueError("intentional validation failure")


class FailingComposite(CompositeElement):
    def __init__(self):
        self.good = ThinLens(f=0.1)
        self.bad = FailingElement(label="Broken child")
        super().__init__()


@dataclass(eq=False, kw_only=True)
class BoundPlane(Plane):
    supplied_bindings: object = None

    @property
    def bindings(self):
        return self.supplied_bindings


@pytest.mark.parametrize("key", [Symbol(("medium", 1)), ("medium", 1)])
def test_make_symbol_creates_independent_instances_with_the_key(key):
    @dataclass
    class Holder:
        symbol: Symbol = make_symbol(key)

    first, second = Holder().symbol, Holder().symbol
    assert first is not key and second is not first
    assert first.key == second.key == ("medium", 1)


def test_variadic_constructor_matches_incremental_registration():
    first, second = Space(d=0.1), ThinLens(f=0.2)
    source = Ray(x=0.001, theta=0.01)
    variadic = System(first, second, optical_input=source)
    incremental = System().add(first).add(second).add_input(source)
    assert variadic.elements == incremental.elements == (first, second)
    assert variadic._inputs[0] is incremental._inputs[0] is source
    assert variadic.ambient_n.value == 1.0 and not variadic.ambient_n.is_variable
    assert System().placements == ()


def test_ambient_parameter_identity_and_rebinding_are_preserved():
    ambient = Parameter(1.33).variable()
    system = System(ambient_n=ambient)
    assert system.ambient_n is system.bindings[AMBIENT_N.key] is ambient
    assert system.set_ambient_n(1.5) is system
    assert system.ambient_n is system.bindings[AMBIENT_N.key]
    assert system.ambient_n.value == 1.5 and not system.ambient_n.is_variable
    assert ambient.value == 1.33 and ambient.is_variable


@pytest.mark.parametrize("value", [0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_ambient_domain_does_not_replace_existing_binding(value):
    system = System()
    original = system.ambient_n
    with pytest.raises(ValueError, match="NaN" if np.isnan(value) else "finite and positive"):
        system.set_ambient_n(value)
    assert system.ambient_n is original and system.bindings[AMBIENT_N.key] is original


@pytest.mark.parametrize("value", [True, 1j, "1.0", object()])
def test_ambient_rejects_non_real_scalars(value):
    with pytest.raises(TypeError, match="real scalar"):
        System(ambient_n=value)


@pytest.mark.parametrize("container", [list, tuple, iter], ids=["list", "tuple", "iterator"])
def test_add_nested_sequences_preserves_composites_and_leaf_order(container):
    first, last = Plane(), ThinLens(f=0.1)
    slab = Slab(d=0.01, n=1.5)
    system = System()
    assert system.add(container([first, [slab, last]])) is system
    assert system.elements == (first, slab, last)
    assert all(p.element is e for p, e in zip(system.placements, (first, slab, last)))


@pytest.mark.parametrize("index", [0, 1, -1, -100, 100])
@pytest.mark.parametrize("count", [1, 2])
def test_indexed_insertion_preserves_python_list_semantics(index, count):
    """Retain the promised Python-list insertion boundary cases as regressions."""
    original = [Space(d=0.1), ThinLens(f=0.1), Space(d=0.2)]
    additions = [ThinLens(f=0.2), ThinLens(f=0.3)][:count]
    position = min(len(original), max(0, index if index >= 0 else len(original) + index))
    expected = original.copy()
    expected[position:position] = additions
    system = System(*original)
    assert system.add(*additions, index=index) is system
    assert len(system.elements) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(system.elements, expected))


def test_indexed_nested_insert_and_keyword_position_keep_order():
    first, second, third, fourth = (ThinLens(f=f) for f in (0.1, 0.2, 0.3, 0.4))
    system = System(first, fourth)
    system.add([second, [third]], index=np.int64(1))
    assert system.elements == (first, second, third, fourth)
    image = Plane()
    system.add(image, z=0.5, index=2)
    assert system.placements[2].element is image and system.placements[2].z == 0.5


@pytest.mark.parametrize("index", [True, np.bool_(True), 1.5, "1", object()])
def test_invalid_insertion_index_does_not_mutate_system(index):
    system = System(Plane())
    before = system.placements
    with pytest.raises(TypeError, match="integer"):
        system.add(ThinLens(f=0.1), index=index)
    assert system.placements == before


@pytest.mark.parametrize("invalid", [None, True, 1.0, object(), "lens", b"lens"])
def test_batch_registration_validates_every_element_before_mutating(invalid):
    system = System(Plane())
    before = system.placements
    with pytest.raises(TypeError, match="OpticalElement"):
        system.add(ThinLens(f=0.1), [invalid])
    assert system.placements == before


def test_empty_registration_and_multiple_absolute_positions_fail_atomically():
    system = System()
    with pytest.raises(ValueError, match="At least one"):
        system.add([])
    with pytest.raises(ValueError, match="absolute position"):
        system.add(ThinLens(f=0.1), Space(d=0.1), z=0.5)
    assert system.placements == ()


@pytest.mark.parametrize("z, error", [
    (True, TypeError), (1j, TypeError), ("0.5", TypeError), (object(), TypeError),
    (-0.1, ValueError), (np.inf, ValueError), (-np.inf, ValueError), (np.nan, ValueError),
])
def test_invalid_absolute_position_is_rejected_at_registration(z, error):
    system = System(Plane())
    before = system.placements
    with pytest.raises(error, match="z"):
        system.add(ThinLens(f=0.1), z=z)
    assert system.placements == before


@pytest.mark.parametrize("position", [Parameter(0.5).variable(), Parameter(0.1) + 0.4])
def test_symbolic_absolute_position_keeps_its_declarative_dependency(position):
    assert System().add(Plane(), z=position).placements[0].z is position


def test_registration_records_call_site_and_repeated_execution_count():
    system = System()
    for _ in range(3):
        system.add(ThinLens(f=0.1))
    sites = [placement.call_site for placement in system.placements]
    assert all(site.file == test_registration_records_call_site_and_repeated_execution_count.__code__.co_filename
               for site in sites)
    assert len({site.line for site in sites}) == 1 and sites[0].line > 0
    assert [site.call_index for site in sites] == [0, 1, 2]
    assert str(sites[0]) == f"{sites[0].file}:{sites[0].line}"


def test_placement_description_includes_identity_and_location():
    placement = Placement(ThinLens(f=0.1, label="Focus"), call_site=CallSite("design.py", 42, 0))
    text = placement.describe(2)
    assert "#2" in text and "ThinLens" in text and "Focus" in text and "design.py:42" in text


def test_add_input_rejects_numerical_states_without_mutating_system():
    system = System()
    numerical = Ray(x=0.001, theta=0.0, n=1.0).numeric()

    with pytest.raises(TypeError, match="declarative optical input"):
        system.add_input(numerical)
    with pytest.raises(TypeError, match="declarative optical input"):
        System(optical_input=numerical)

    assert system.add_input(Ray(x=0.001, theta=0.0, n=1.0)) is system


def test_add_input_rejects_invalid_and_additional_inputs_without_replacement():
    system = System()
    with pytest.raises(TypeError, match="ParaxialState"):
        system.add_input(object())
    with pytest.raises(ValueError, match="At least one"):
        system.add_input([])
    first = Ray(x=0, theta=0)
    assert system.add_input([first]) is system
    with pytest.raises(ValueError, match="Multiple"):
        system.add_input(Ray(x=0.001, theta=0))
    assert system._inputs == [first]


@pytest.mark.parametrize("key", ["", None, 123, False, ("calibration", 1)])
def test_bind_accepts_hashable_keys_and_wraps_real_values(key):
    system = System()
    assert system.bind(key, 293.15) is system
    node = system.bindings[key]
    assert isinstance(node, Parameter) and not node.is_variable
    assert node.value == 293.15 and node.name == (key if isinstance(key, str) else None)


def test_bind_preserves_nodes_complex_values_and_a_read_only_live_view():
    system = System()
    parameter = Parameter(2.0).variable()
    view = system.bindings
    system.bind("gain", parameter).bind("phase", 1 + 2j)
    assert view["gain"] is parameter
    assert isinstance(view["phase"], Literal) and view["phase"].value == 1 + 2j
    with pytest.raises(TypeError):
        view["gain"] = Parameter(9.0)
    system.bind("gain", 3.0)
    assert view["gain"].value == 3.0 and parameter.value == 2.0


@pytest.mark.parametrize("key, value", [([], 1), ({}, 1), ("invalid", True),
                                          ("invalid", [1]), ("invalid", "1"), ("invalid", object())])
def test_invalid_binding_leaves_existing_bindings_unchanged(key, value):
    system = System()
    before = dict(system.bindings)
    with pytest.raises(TypeError):
        system.bind(key, value)
    assert system.bindings.keys() == before.keys()
    assert all(system.bindings[key] is node for key, node in before.items())


def test_collect_bindings_combines_sources_and_deduplicates_reused_elements():
    first = BoundPlane(supplied_bindings={"calibration": Parameter(2)})
    second = BoundPlane(supplied_bindings={("phase", 0): Literal(1j)})
    source = GaussianBeam(q=0.02j, wavelength=1.064e-6, n=1.0)
    system = System(first, second, first, optical_input=source).bind("user", 3)
    combined = system._collect_bindings()
    assert combined["wavelength"] is source.wavelength
    assert combined["calibration"] is first.bindings["calibration"]
    assert combined[("phase", 0)] is second.bindings[("phase", 0)]
    assert combined["user"] is system.bindings["user"]
    combined["new"] = 1
    assert "new" not in first.bindings and "new" not in system.bindings


def test_collect_bindings_rejects_collisions_instead_of_overriding():
    shared = Parameter(2)
    first = BoundPlane(supplied_bindings={"gain": shared})
    second = BoundPlane(supplied_bindings={"gain": shared})
    for system in (System(first, second), System(first).bind("gain", shared)):
        with pytest.raises(ValueError, match="gain"):
            system._collect_bindings()
    assert first.bindings["gain"] is shared


def test_system_binding_cannot_silently_override_source_wavelength():
    source = GaussianBeam(q=0.02j, wavelength=1.064e-6, n=1.0)
    system = System(optical_input=source).bind("wavelength", 1.55e-6)
    with pytest.raises(ValueError, match="Duplicate symbol binding.*wavelength"):
        system._collect_bindings()
    assert source.wavelength.value == 1.064e-6
    assert system.bindings["wavelength"].value == 1.55e-6


def test_constraints_collect_system_input_composite_and_children_once_in_order():
    source = Ray(x=0, theta=0, n=1.0)
    slab = Slab(d=0.01, n=1.5)
    direct, input_constraint, composite_constraint = source.x >= -1, source.theta <= 1, slab.d <= 0.02
    system = System(slab, slab, optical_input=source).require(direct)
    source.require(input_constraint)
    slab.require(composite_constraint)
    expected = (direct, input_constraint, composite_constraint,
                *slab.front.constraints, *slab.body.constraints, *slab.back.constraints)
    actual = system.constraints
    assert isinstance(actual, tuple) and len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))
    assert system.system_constraints == (direct,) and slab.constraints == (composite_constraint,)


def test_validation_aggregates_missing_input_and_nested_element_errors():
    system = System(FailingElement(label="Broken direct"), FailingComposite())
    with pytest.raises(SystemValidationError) as error:
        system._validate()
    assert len(error.value.errors) == 3
    text = str(error.value)
    assert "No input" in text and "Broken direct" in text and "child 'bad'" in text
    assert "intentional validation failure" in text and "Placement #0" in text


def test_structural_equality_is_rejected_before_batch_mutation():
    @dataclass(kw_only=True)
    class EqualPlane(Plane):
        pass

    system = System()
    with pytest.raises(TypeError, match=r"dataclass\(eq=False\)"):
        system.add(Plane(), EqualPlane())
    assert system.placements == ()


def test_summary_describes_design_without_resolving_or_mutating_it():
    lens = ThinLens(f=0.1, label="Focus").variable("f")
    system = System(lens, optical_input=Ray(x=0, theta=0, n=1.0))
    constraint = lens.f >= 0.05
    assert system.require(constraint) is system
    before = system.placements
    text = str(system)
    assert "Focus" in text and "ThinLens" in text and "VAR" in text
    assert "Constraints: 1" in text and "Ray" in text
    assert system.placements == before and system.system_constraints[0] is constraint


def test_variadic_constructor_records_user_call_site_and_build_error_location():
    import inspect
    line = inspect.currentframe().f_lineno + 1
    system = System(ThinLens(f=0), Plane(), optical_input=Ray(x=0, theta=0, n=1))
    sites = [placement.call_site for placement in system.placements]
    filename = test_variadic_constructor_records_user_call_site_and_build_error_location.__code__.co_filename
    assert all(site.file == filename and site.line == line for site in sites)
    assert all(site.call_index == 0 for site in sites)
    with pytest.raises(ValueError) as error:
        system.build()
    assert f"{filename}:{line}" in str(error.value)


# ------------------------
# REFERENCE POINT (at=)
# ------------------------
def test_at_requires_an_absolute_position():
    system = System()
    with pytest.raises(ValueError, match="at requires"):
        system.add(Space(d=0.1), at="center")
    assert system.placements == ()


def test_the_placement_stores_the_position_of_the_front_after_applying_at():
    from diffractix.composites import Telescope
    gap = Space(d=0.04)
    telescope = Telescope(f1=0.1, f2=0.2)
    system = System().add(gap, z=0.1).add(gap, z=0.1, at="center").add(gap, z=0.1, at="back")
    system.add(gap, z=0.1, at=0.01).add(telescope, z=0.5, at=telescope.lens2)
    fronts = [placement.z for placement in system.placements]
    assert fronts[0] == 0.1 and isinstance(fronts[0], float)   # no at: untouched
    # A Space length is a graph value, so the shifted front is a live expression.
    values = [front.evaluate() if hasattr(front, "evaluate") else front for front in fronts]
    np.testing.assert_allclose(values[1:], [0.08, 0.06, 0.09, 0.2], atol=1e-12)


@pytest.mark.parametrize("at, error, message", [
    (True, TypeError, "at must be"),
    (object(), TypeError, "at must be"),
    ("middle", ValueError, "at must be one of"),
    (("front",), TypeError, "at must be"),
    ((Space(d=0.1), 0), TypeError, "at must be"),
    (float("nan"), ValueError, "finite"),
    (float("inf"), ValueError, "finite"),
    (Space(d=0.1), ValueError, "not part of"),
], ids=["bool", "other-type", "unknown-name", "tuple", "element-with-occurrence", "nan", "inf", "foreign-element"])
def test_invalid_at_is_rejected_without_mutating_the_system(at, error, message):
    system = System(Plane())
    before = system.placements
    with pytest.raises(error, match=message):
        system.add(ThinLens(f=0.1), z=0.2, at=at)
    assert system.placements == before


class Group(CompositeElement):
    def __init__(self, **children):
        for name, child in children.items():
            setattr(self, name, child)
        super().__init__()


def front_of(placement):
    return placement.z.evaluate() if hasattr(placement.z, "evaluate") else placement.z


def test_a_child_reference_is_the_length_in_front_of_it_however_deeply_nested():
    inner = Group(b=Space(d=0.2), c=Space(d=0.3))
    outer = Group(a=Space(d=0.1), inner=inner)
    system = System()
    targets = (outer.a, inner.b, inner.c, inner)   # the last is a nested composite: its first leaf
    for target in targets:
        system.add(outer, z=1.0, at=target)
    np.testing.assert_allclose([front_of(p) for p in system.placements], [1.0, 0.9, 0.7, 0.9], atol=1e-12)


def test_a_child_reference_follows_variable_lengths_and_skips_zero_length_children():
    distance = Parameter(0.1).variable()
    group = Group(a=Space(d=distance), lens=ThinLens(f=0.1), b=Space(d=2 * distance))
    system = System().add(group, z=1.0, at=group.lens).add(group, z=1.0, at=group.b)
    assert [front_of(p) for p in system.placements] == pytest.approx([0.9, 0.9])
    distance.value = 0.2
    assert [front_of(p) for p in system.placements] == pytest.approx([0.8, 0.8])


def test_repeated_references_are_ambiguous():
    shared = Space(d=0.01)
    group = Group(first=shared, middle=Space(d=0.05), last=shared)
    inner = Group(x=Space(d=0.02))
    outer = Group(a=inner, b=Space(d=0.03), c=inner)   # a repeated composite repeats its first leaf
    system = System()
    for container, target in ((group, shared), (outer, inner)):
        with pytest.raises(ValueError, match="more than once"):
            system.add(container, z=0.5, at=target)
    assert system.placements == ()
    assert front_of(system.add(group, z=0.5, at=group.middle).placements[0]) == pytest.approx(0.49)


@pytest.mark.parametrize("make_added, make_reference", [
    (lambda: Group(a=Space(d=0.1)), lambda added: Space(d=0.1)),          # foreign element
    (lambda: Group(a=Space(d=0.1)), lambda added: Group()),               # empty composite
    (lambda: ThinLens(f=0.1), lambda added: Space(d=0.1)),                # plain element has no inside
], ids=["foreign", "empty-composite", "plain-element"])
def test_references_that_are_not_inside_the_added_element_are_rejected(make_added, make_reference):
    added = make_added()
    system = System()
    with pytest.raises(ValueError, match="not part of"):
        system.add(added, z=0.5, at=make_reference(added))
    assert system.placements == ()


def test_an_element_or_composite_cannot_be_its_own_reference():
    from diffractix.composites import Telescope
    gap, telescope = Space(d=0.1), Telescope(f1=0.1, f2=0.2)
    for element in (gap, telescope):
        with pytest.raises(ValueError, match="not part of"):
            System().add(element, z=0.2, at=element)


def test_at_accepts_symbolic_lengths_and_keeps_them_live():
    reference = Parameter(0.01).variable()
    system = System().add(Space(d=0.1), z=0.3, at=reference + 0.01)
    front = system.placements[0].z
    assert front.evaluate() == pytest.approx(0.28)
    reference.value = 0.05
    assert front.evaluate() == pytest.approx(0.24)


@pytest.mark.parametrize("z", [0.0, -0.0, 0.2])
def test_nonnegative_absolute_positions_are_stored_unchanged(z):
    assert System().add(Plane(), z=z).placements[0].z == z


@pytest.mark.parametrize("z", [-1e-17, -1e-9, -0.1])
def test_negative_absolute_positions_are_rejected_however_small(z):
    system = System()
    with pytest.raises(ValueError, match="cannot be negative"):
        system.add(Plane(), z=z)
    assert system.placements == ()


def test_shared_provider_in_two_composites_of_one_system_contributes_once():
    from diffractix.composites import CompositeElement

    class Pair(CompositeElement):
        def __init__(self, shared):
            super().__init__()
            self.gap = Space(d=0.01)
            self.shared = shared

    shared = BoundPlane(supplied_bindings={"calibration": 1.5})
    system = System(Pair(shared), Pair(shared), optical_input=Ray(x=0, theta=0, n=1))
    assert system.build().bindings["calibration"] == 1.5


def test_summary_marks_variable_fixed_derived_and_empty_parameters():
    distance = Parameter(0.1).variable()
    system = System(
        Space(d=distance, label="Gap"),                 # variable length, inherited index
        ThinLens(f=2 * distance, label="Derived"),      # an expression of the variable
        ThinLens(f=0.25, label="Fixed"),
        optical_input=Ray(x=0, theta=0, n=1.0),
    )
    rows = {line.split()[3]: line for line in str(system).splitlines() if "relative" in line}

    assert "d=0.1 [VAR]" in rows["Gap"] and "n=-" in rows["Gap"]
    assert "[EXPR]" in rows["Derived"] and "[VAR]" not in rows["Derived"]
    assert "f=0.25 [FIX]" in rows["Fixed"]


def test_summary_names_constraints_by_label_then_function_name_then_expression():
    distance = Parameter(0.1, name="distance").variable()
    system = System(Space(d=distance), optical_input=Ray(x=0, theta=0, n=1.0))

    def stay_short():
        return distance <= 1

    system.require(distance <= 2, label="Hard stop")
    system.require(stay_short)
    system.require(distance >= 0.01)
    text = str(system)

    listed = text.split("Constraints\n\n")[1].splitlines()
    assert [line.split(maxsplit=1)[1] for line in listed[:3]] == [
        "Hard stop", "stay_short", "distance=0.1[V] >= 0.01"]
    assert "Constraints: 4" in text    # the three above and the Space's own d >= 0
