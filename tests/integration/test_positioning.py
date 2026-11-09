"""Absolute placement: which point of an element sits at z (`at=`), and float tolerance."""
import numpy as np
import pytest

from diffractix import (
    FourF, Parameter, Plane, Ray, Slab, Solver, Space, System, Telescope, ThickLens, ThinLens,
)
from diffractix.backend import grad, jacobian, jit, xp
from diffractix.composites import CompositeElement


def source():
    return Ray(x=0.001, theta=0.01, n=1)


class Group(CompositeElement):
    def __init__(self, **children):
        for name, child in children.items():
            setattr(self, name, child)
        super().__init__()


def span(result, element):
    return result[element].z_start, result[element].z_end


# ---------------
# NAMED POINTS
# ---------------
@pytest.mark.parametrize("at, front", [("front", 0.10), ("center", 0.09), ("back", 0.08)])
def test_named_point_of_an_element_sits_at_z(at, front):
    gap = Space(d=0.02)
    result = System(optical_input=source()).add(gap, z=0.10, at=at).build().run()
    np.testing.assert_allclose(span(result, gap), (front, front + 0.02), atol=1e-12)


def test_front_is_the_default_reference():
    first, second = Space(d=0.02), Space(d=0.02)
    default = System(optical_input=source()).add(first, z=0.1).build().run()
    explicit = System(optical_input=source()).add(second, z=0.1, at="front").build().run()
    assert span(default, first) == span(explicit, second)


@pytest.mark.parametrize("at", ["front", "center", "back"])
def test_zero_length_elements_have_one_position(at):
    lens = ThinLens(f=0.1)
    result = System(optical_input=source()).add(lens, z=0.1, at=at).build().run()
    assert span(result, lens) == (pytest.approx(0.1), pytest.approx(0.1))


@pytest.mark.parametrize("at, front", [("front", 0.1), ("center", 0.095), ("back", 0.09)])
def test_composite_reference_points_use_the_whole_composite_length(at, front):
    lens = ThickLens(d=0.01, n=1.5, R1=0.2, R2=-0.2)
    result = System(optical_input=source()).add(lens, z=0.1, at=at).build().run()
    np.testing.assert_allclose(
        (result[lens.front].z_start, result[lens.back].z_end), (front, front + 0.01), atol=1e-12)


# ----------------------------
# ELEMENTS INSIDE A COMPOSITE
# ----------------------------
@pytest.mark.parametrize("reference, distance_from_front", [("lens1", 0.0), ("space", 0.0), ("lens2", 0.3)])
def test_a_child_element_sits_at_z_with_the_rest_of_the_composite_around_it(reference, distance_from_front):
    telescope = Telescope(f1=0.1, f2=0.2)
    child = getattr(telescope, reference)
    result = System(optical_input=source()).add(telescope, z=0.4, at=child).build().run()
    front = 0.4 - distance_from_front
    assert result[child].z_start == pytest.approx(0.4, abs=1e-12)
    np.testing.assert_allclose(
        [result[telescope.lens1].z_start, result[telescope.lens2].z_start], [front, front + 0.3], atol=1e-12)


def test_nested_children_and_inner_reference_points():
    inner = Group(a=Space(d=0.02), b=ThinLens(f=0.1), c=Space(d=0.03))
    outer = Group(first=Space(d=0.01), inner=inner)
    # c starts 0.01 + 0.02 after the front of outer.
    result = System(optical_input=source()).add(outer, z=0.2, at=inner.c).build().run()
    assert span(result, inner.c)[0] == pytest.approx(0.2)
    assert span(result, outer.first)[0] == pytest.approx(0.17)
    # The centre of c at 0.2: put c's front half its length earlier.
    centered = System(optical_input=source()).add(
        outer, z=0.2 - inner.c.element_length / 2, at=inner.c).build().run()
    assert sum(span(centered, inner.c)) / 2 == pytest.approx(0.2)


def test_a_repeated_child_cannot_be_a_reference_but_others_can():
    shared = Space(d=0.01)
    middle = Space(d=0.05)
    group = Group(first=shared, middle=middle, last=shared)
    with pytest.raises(ValueError, match="more than once"):
        System(optical_input=source()).add(group, z=0.3, at=shared)
    result = System(optical_input=source()).add(group, z=0.3, at=middle).build().run()
    assert span(result, middle)[0] == pytest.approx(0.3)


# ----------------------------
# ARBITRARY REFERENCE LENGTHS
# ----------------------------
def test_any_length_expression_from_the_front_can_be_the_reference():
    gap = Space(d=0.07)
    result = System(optical_input=source()).add(gap, z=0.2, at=gap.element_length / 3.5).build().run()
    assert span(result, gap)[0] == pytest.approx(0.2 - 0.02)


@pytest.mark.parametrize("at, front", [(0.005, 0.095), (0, 0.1), (-0.01, 0.11)])
def test_plain_numbers_measure_from_the_front_and_may_be_negative(at, front):
    gap = Space(d=0.02)
    result = System(optical_input=source()).add(gap, z=0.1, at=at).build().run()
    assert span(result, gap)[0] == pytest.approx(front)


def test_a_reference_that_puts_the_front_before_the_system_start_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative.*after applying at"):
        System(optical_input=source()).add(Space(d=0.02), z=0.01, at=0.02)
    # With a symbolic position the front is only known at build time.
    system = System(optical_input=source()).add(Space(d=0.02), z=Parameter(0.01), at=0.02)
    with pytest.raises(ValueError, match="unsatisfiable"):
        system.build()


def test_reference_lengths_can_be_built_from_element_lengths():
    lens = ThickLens(d=0.01, n=1.5, R1=0.2, R2=-0.2)
    gap = Space(d=0.04)
    system = System(optical_input=source())
    system.add(lens, z=0.1, at=lens.element_length / 2 + 0.001)
    system.add(gap, z=0.2, at=gap.element_length - 0.01)
    result = system.build().run()
    assert result[lens.front].z_start == pytest.approx(0.1 - 0.005 - 0.001)
    assert span(result, gap)[0] == pytest.approx(0.2 - 0.03)


# ---------------------------------
# VARIABLE LENGTHS AND DERIVATIVES
# ---------------------------------
def test_a_fixed_centre_keeps_it_while_the_thickness_changes():
    thickness = Parameter(0.01).variable(bounds=(0.002, 0.04))
    lens = ThickLens(d=thickness, n=1.5, R1=0.2, R2=-0.2)
    simulation = System(optical_input=source()).add(lens, z=0.1, at="center").build()

    def vertices(theta):
        result = simulation.run(theta)
        return xp.stack((result[lens.front].z_start, result[lens.back].z_end))

    point = xp.array([0.02])
    np.testing.assert_allclose(vertices(point), [0.09, 0.11], atol=1e-12)
    for function in (jacobian(vertices), jit(jacobian(vertices))):
        np.testing.assert_allclose(function(point), [[-0.5], [0.5]], atol=1e-12)


def test_the_reference_offset_can_itself_be_a_design_variable():
    offset = Parameter(0.01).variable(bounds=(0.0, 0.05))
    gap = Space(d=0.02)
    simulation = System(optical_input=source()).add(gap, z=0.1, at=offset).build()
    start = lambda theta: simulation.run(theta)[gap].z_start
    point = xp.array([0.03])
    assert start(point) == pytest.approx(0.07)
    for function in (grad(start), jit(grad(start))):
        np.testing.assert_allclose(function(point), [-1.0])


def test_solver_can_choose_a_thickness_around_a_fixed_centre():
    thickness = Parameter(0.004).variable(bounds=(0.002, 0.04))
    lens = ThickLens(d=thickness, n=1.5, R1=0.2, R2=-0.2)
    system = System(optical_input=source()).add(lens, z=0.1, at="center")
    solver = Solver(system).target(value=lambda sim: sim[lens.back].z_end, target=0.108, scale=0.01)
    solution = solver.solve(options={"ftol": 1e-14})
    assert solution.success and solution.feasible
    assert solution.value(thickness) == pytest.approx(0.016, abs=1e-7)
    assert solution.value(lens.d) == pytest.approx(0.016, abs=1e-7)


def test_later_elements_follow_a_reference_placed_thick_element():
    lens = ThickLens(d=0.01, n=1.5, R1=0.2, R2=-0.2)
    station = Plane(label="station")
    system = System(optical_input=source()).add(lens, z=0.1, at="center").add(station, z=0.2)
    result = system.build().run()
    assert span(result, station)[0] == pytest.approx(0.2)
    assert result[lens.back].z_end == pytest.approx(0.105)


# ------------------------------
# NEGATIVE FRONTS AND OVERLAPS
# ------------------------------
def test_a_station_at_the_exact_sum_of_earlier_gaps_builds():
    station = Plane()
    system = System(Space(d=0.125), Space(d=0.25), optical_input=source()).add(station, z=0.375)
    result = system.build().run()
    assert span(result, station)[0] == 0.375


def test_a_real_overlap_is_still_infeasible():
    system = System(Space(d=0.1), Space(d=0.2), optical_input=source()).add(Plane(), z=0.29)
    with pytest.raises(ValueError, match="unsatisfiable"):
        system.build()


def test_generated_gaps_are_strictly_non_negative():
    # Variable lengths keep their constraints; satisfied constant ones are dropped at build.
    distance = Parameter(0.1).variable(bounds=(0.01, 0.15))
    system = System(Space(d=distance), optical_input=source()).add(Plane(), z=0.2)
    simulation = system.build()
    assert [entry.lower_bound for entry in simulation.constraint_descriptors] == [0, 0]


def test_touching_thick_lenses_build_exactly():
    first = ThickLens(d=0.0123, n=1.5, R1=0.2, R2=-0.2)
    second = ThickLens(d=0.0123, n=1.5, R1=0.2, R2=-0.2)
    system = System(optical_input=source()).add(first, z=0.1).add(second, z=0.1 + 0.0123)
    result = system.build().run()
    assert result[second.front].z_start == pytest.approx(result[first.back].z_end, abs=1e-12)


def test_four_f_relay_can_be_pinned_by_its_fourier_plane():
    relay = FourF(f1=0.1, f2=0.2)
    result = System(optical_input=source()).add(relay, z=0.5, at=relay.fourier_space).build().run()
    # fourier_space starts after input_space (f1).
    assert result[relay.fourier_space].z_start == pytest.approx(0.5)
    assert result[relay.input_space].z_start == pytest.approx(0.5 - 0.1)


def test_slab_can_be_centred_in_a_fixed_length_module():
    """The multistation pattern: a crystal centred between fixed module faces."""
    module_start, module_end = 0.2, 0.215
    crystal = Slab(d=Parameter(0.009).variable(bounds=(0.006, 0.014)), n=1.45)
    simulation = System(optical_input=source()).add(
        crystal, z=(module_start + module_end) / 2, at="center").build()
    for thickness in (0.006, 0.009, 0.014):
        guess = simulation.initial_values
        guess[crystal.d] = thickness
        result = simulation.run(guess.values)
        air = (module_end - module_start - thickness) / 2
        assert result[crystal.front].z_start == pytest.approx(module_start + air)
        assert result[crystal.back].z_end == pytest.approx(module_end - air)
