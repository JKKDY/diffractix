import pytest

from autograd import grad
import autograd.numpy as np

from diffractix.composites import Slab, ThickLens
from diffractix.elements import Interface, Space, ThinLens
from diffractix.graph import Parameter
from diffractix.system.system import System
from diffractix.beams import GaussianBeam


def create_beam(n=1.0):
    return GaussianBeam.from_waist(
        w0=1e-3,
        wavelength=1064e-9,
        n=n,
    )


def theta_with(simulation, *updates):
    theta = simulation.initial_values.copy()

    for parameter, value in updates:
        index = next(
            info.index
            for info in simulation.parameter_info
            if info.parameter is parameter
        )
        theta[index] = value

    return theta


def propagate_interface(q, n1, n2, R):
    if np.isinf(R):
        C = 0.0
    else:
        C = (n1 - n2) / (R * n2)

    D = n1 / n2

    return q / (C * q + D)




# ----------
# COMPOSITES
# ----------

def test_slab_builds_as_three_simulation_steps():
    beam = create_beam()
    slab = Slab(
        d=0.02,
        n=1.5,
    )

    system = System()
    system.add_input_beam(beam)
    system.add(slab)

    simulation = system.build()
    result = simulation.run()

    assert len(simulation.steps) == 3
    assert len(result.states) == 4
    assert np.allclose(
        result.z,
        np.array([0.0, 0.0, 0.02, 0.02]),
    )


def test_slab_enters_and_exits_material_medium():
    beam = create_beam(n=1.0)
    slab = Slab(
        d=0.02,
        n=1.5,
    )

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(slab)

    result = system.build().run()

    assert result.states[0].n == pytest.approx(1.0)
    assert result.states[1].n == pytest.approx(1.5)
    assert result.states[2].n == pytest.approx(1.5)
    assert result.states[3].n == pytest.approx(1.0)


def test_slab_matches_equivalent_explicit_system():
    first_beam = create_beam()
    second_beam = create_beam()

    slab = Slab(
        d=0.02,
        n=1.5,
    )

    composite_system = System(ambient_n=1.0)
    composite_system.add_input_beam(first_beam)
    composite_system.add(slab)

    explicit_system = System(ambient_n=1.0)
    explicit_system.add_input_beam(second_beam)
    explicit_system.add(Interface(
        n1=1.0,
        n2=1.5,
        R=np.inf,
    ))
    explicit_system.add(Space(
        d=0.02,
        n=1.5,
    ))
    explicit_system.add(Interface(
        n1=1.5,
        n2=1.0,
        R=np.inf,
    ))

    composite = composite_system.build().run()
    explicit = explicit_system.build().run()

    assert np.allclose(composite.z, explicit.z)
    assert composite.final.q == pytest.approx(explicit.final.q)
    assert composite.final.n == pytest.approx(explicit.final.n)


def test_variable_slab_thickness_controls_simulation_length():
    beam = create_beam()
    slab = Slab(
        d=0.02,
        n=1.5,
    ).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(slab)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.05]))

    assert result.z[-1] == pytest.approx(0.05)


def test_variable_slab_index_is_single_shared_dimension():
    beam = create_beam()
    slab = Slab(
        d=0.02,
        n=1.5,
    ).variable("n")

    system = System()
    system.add_input_beam(beam)
    system.add(slab)

    simulation = system.build()

    assert len(simulation.initial_values) == 1


def test_variable_slab_index_updates_all_internal_media():
    beam = create_beam()
    slab = Slab(
        d=0.02,
        n=1.5,
    ).variable("n")

    system = System()
    system.add_input_beam(beam)
    system.add(slab)

    simulation = system.build()
    result = simulation.run(np.array([1.8]))

    assert result.states[1].n == pytest.approx(1.8)
    assert result.states[2].n == pytest.approx(1.8)
    assert result.states[3].n == pytest.approx(1.0)


def test_thick_lens_builds_as_three_simulation_steps():
    beam = create_beam()

    lens = ThickLens(
        d=0.01,
        n=1.5,
        R1=0.1,
        R2=-0.1,
    )

    system = System()
    system.add_input_beam(beam)
    system.add(lens)

    simulation = system.build()
    result = simulation.run()

    assert len(simulation.steps) == 3
    assert len(result.states) == 4
    assert result.z[-1] == pytest.approx(0.01)


def test_thick_lens_matches_equivalent_explicit_system():
    first_beam = create_beam()
    second_beam = create_beam()

    lens = ThickLens(
        d=0.01,
        n=1.5,
        R1=0.1,
        R2=-0.08,
    )

    composite_system = System(ambient_n=1.0)
    composite_system.add_input_beam(first_beam)
    composite_system.add(lens)

    explicit_system = System(ambient_n=1.0)
    explicit_system.add_input_beam(second_beam)
    explicit_system.add(Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    ))
    explicit_system.add(Space(
        d=0.01,
        n=1.5,
    ))
    explicit_system.add(Interface(
        n1=1.5,
        n2=1.0,
        R=-0.08,
    ))

    composite = composite_system.build().run()
    explicit = explicit_system.build().run()

    assert np.allclose(composite.z, explicit.z)
    assert composite.final.q == pytest.approx(explicit.final.q)
    assert composite.final.n == pytest.approx(explicit.final.n)


def test_variable_thick_lens_radii_are_independent_dimensions():
    beam = create_beam()

    lens = ThickLens(
        d=0.01,
        n=1.5,
        R1=0.1,
        R2=-0.1,
    ).variable("R1", "R2")

    system = System()
    system.add_input_beam(beam)
    system.add(lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 2



# ---------------------
# DEPENDENT EXPRESSIONS
# ---------------------

def test_space_length_can_depend_on_lens_focal_length():
    beam = create_beam()
    lens = ThinLens(f=0.2).variable("f")
    space = Space(d=2 * lens.f)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(space)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.3]))

    assert result.z[-1] == pytest.approx(0.6)


def test_dependent_expression_uses_same_theta_as_source_parameter():
    beam = create_beam()
    lens = ThinLens(f=0.2).variable("f")
    space = Space(d=2 * lens.f)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(space)

    simulation = system.build()
    result = simulation.run(np.array([0.4]))

    q1 = beam.q / (1.0 - beam.q / 0.4)
    expected = q1 + 0.8

    assert result.final.q == pytest.approx(expected)


def test_dependency_can_reference_downstream_element():
    beam = create_beam()

    lens = ThinLens(f=0.2).variable("f")
    space = Space(d=2 * lens.f)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    simulation = system.build()
    result = simulation.run(np.array([0.3]))

    q1 = beam.q + 0.6
    expected = q1 / (1.0 - q1 / 0.3)

    assert result.z[-1] == pytest.approx(0.6)
    assert result.final.q == pytest.approx(expected)


def test_chained_cross_element_dependencies_have_one_independent_parameter():
    beam = create_beam()

    first_lens = ThinLens(
        f=0.1,
        label="First",
    ).variable("f")

    second_lens = ThinLens(
        f=2 * first_lens.f,
        label="Second",
    )

    space = Space(
        d=first_lens.f + second_lens.f,
    )

    system = System()
    system.add_input_beam(beam)
    system.add(first_lens)
    system.add(space)
    system.add(second_lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.2]))

    assert result.z[-1] == pytest.approx(0.6)


def test_chained_dependency_updates_all_derived_values():
    beam = create_beam()

    first_lens = ThinLens(f=0.1).variable("f")
    second_lens = ThinLens(f=2 * first_lens.f)
    space = Space(d=first_lens.f + second_lens.f)

    system = System()
    system.add_input_beam(beam)
    system.add(first_lens)
    system.add(space)
    system.add(second_lens)

    simulation = system.build()
    result = simulation.run(np.array([0.2]))

    q1 = beam.q / (1.0 - beam.q / 0.2)
    q2 = q1 + 0.6
    expected = q2 / (1.0 - q2 / 0.4)

    assert result.final.q == pytest.approx(expected)


def test_dependent_expression_is_differentiable():
    beam = create_beam()

    lens = ThinLens(f=0.2).variable("f")
    space = Space(d=2 * lens.f)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(space)

    simulation = system.build()

    def objective(theta):
        return simulation.run(theta).z[-1]

    derivative = grad(objective)(simulation.initial_values)

    assert derivative[0] == pytest.approx(2.0)



# ---------------------------
# VARIABLE ABSOLUTE POSITIONS
# ---------------------------

def test_variable_absolute_position_controls_generated_space():
    beam = create_beam()

    position = Parameter(
        0.2,
        name="position",
    ).variable()

    lens = ThinLens(f=0.5)

    system = System()
    system.add_input_beam(beam)
    system.add(lens, z=position)

    simulation = system.build()
    result = simulation.run(np.array([0.4]))

    assert np.allclose(
        result.z,
        np.array([0.0, 0.4, 0.4]),
    )


def test_absolute_position_can_depend_on_element_parameter():
    beam = create_beam()

    first_lens = ThinLens(f=0.2).variable("f")
    second_lens = ThinLens(f=0.5)

    system = System()
    system.add_input_beam(beam)
    system.add(first_lens)
    system.add(second_lens, z=3 * first_lens.f)

    simulation = system.build()
    result = simulation.run(np.array([0.3]))

    assert np.allclose(
        result.z,
        np.array([0.0, 0.0, 0.9, 0.9]),
    )


def test_multiple_absolute_positions_can_share_one_parameter():
    beam = create_beam()

    position = Parameter(
        0.2,
        name="position",
    ).variable()

    first = ThinLens(f=0.5)
    second = ThinLens(f=0.5)

    system = System()
    system.add_input_beam(beam)
    system.add(first, z=position)
    system.add(second, z=3 * position)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.3]))

    assert np.allclose(
        result.z,
        np.array([
            0.0,
            0.3,
            0.3,
            0.9,
            0.9,
        ]),
    )


def test_variable_upstream_length_changes_generated_absolute_gap():
    beam = create_beam()

    space = Space(d=0.1).variable("d")
    position = Parameter(
        0.5,
        name="position",
    ).variable()

    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens, z=position)

    simulation = system.build()

    theta = theta_with(
        simulation,
        (space.d.node, 0.2),
        (position, 0.7),
    )

    result = simulation.run(theta)

    assert np.allclose(
        result.z,
        np.array([
            0.0,
            0.2,
            0.7,
            0.7,
        ]),
    )


def test_absolute_position_dependency_is_differentiable():
    beam = create_beam()

    position = Parameter(
        0.2,
        name="position",
    ).variable()

    lens = ThinLens(f=0.5)

    system = System()
    system.add_input_beam(beam)
    system.add(lens, z=2 * position)

    simulation = system.build()

    def objective(theta):
        return simulation.run(theta).z[-1]

    derivative = grad(objective)(simulation.initial_values)

    assert derivative[0] == pytest.approx(2.0)


# ----------------------
# VARIABLE AMBIENT INDEX
# ----------------------

def test_variable_ambient_index_is_simulation_parameter():
    beam = create_beam(n=1.0)

    system = System(
        ambient_n=1.0,
        ambient_n_variable=True,
    )
    system.add_input_beam(beam)
    system.add(Space(d=0.2))

    simulation = system.build()

    assert len(simulation.initial_values) == 1
    assert simulation.parameter_info[0].parameter is system.ambient_n
    assert simulation.initial_values[0] == pytest.approx(1.0)


def test_variable_ambient_index_updates_inherited_space_medium():
    beam = create_beam(n=1.0)

    system = System(
        ambient_n=1.0,
        ambient_n_variable=True,
    )
    system.add_input_beam(beam)
    system.add(Space(d=0.2))

    simulation = system.build()
    result = simulation.run(np.array([1.4]))

    assert result.final.n == pytest.approx(1.4)


def test_variable_ambient_index_is_shared_by_multiple_inherited_elements():
    beam = create_beam(n=1.0)

    system = System(
        ambient_n=1.0,
        ambient_n_variable=True,
    )
    system.add_input_beam(beam)
    system.add(Space(d=0.1))
    system.add(ThinLens(f=0.3))
    system.add(Space(d=0.2))

    simulation = system.build()
    result = simulation.run(np.array([1.6]))

    assert len(simulation.initial_values) == 1
    assert result.states[1].n == pytest.approx(1.6)
    assert result.states[2].n == pytest.approx(1.6)
    assert result.states[3].n == pytest.approx(1.6)


def test_variable_ambient_and_element_parameter_are_independent():
    beam = create_beam(n=1.0)

    system = System(
        ambient_n=1.0,
        ambient_n_variable=True,
    )

    space = Space(d=0.2).variable("d")

    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    assert len(simulation.initial_values) == 2

    theta = theta_with(
        simulation,
        (system.ambient_n, 1.5),
        (space.d.node, 0.4),
    )

    result = simulation.run(theta)

    assert result.final.n == pytest.approx(1.5)
    assert result.z[-1] == pytest.approx(0.4)


def test_variable_ambient_can_drive_interface_input_index():
    beam = create_beam(n=1.0)

    system = System(
        ambient_n=1.0,
        ambient_n_variable=True,
    )

    interface = Interface(
        n1=system.ambient_n,
        n2=1.5,
        R=0.1,
    )

    system.add_input_beam(beam)
    system.add(interface)

    simulation = system.build()

    first = simulation.run(np.array([1.0]))
    second = simulation.run(np.array([1.2]))

    assert first.final.n == pytest.approx(1.5)
    assert second.final.n == pytest.approx(1.5)
    assert not np.isclose(first.final.q, second.final.q)



# -----------------
# CURVED INTERFACES
# -----------------

def test_positive_curved_interface_matches_abcd_formula():
    beam = create_beam(n=1.0)

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    )

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)

    result = system.build().run()

    expected = propagate_interface(
        beam.q,
        n1=1.0,
        n2=1.5,
        R=0.1,
    )

    assert result.final.q == pytest.approx(expected)
    assert result.final.n == pytest.approx(1.5)


def test_negative_curved_interface_matches_abcd_formula():
    beam = create_beam(n=1.0)

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=-0.1,
    )

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)

    result = system.build().run()

    expected = propagate_interface(
        beam.q,
        n1=1.0,
        n2=1.5,
        R=-0.1,
    )

    assert result.final.q == pytest.approx(expected)


def test_opposite_interface_curvatures_produce_different_states():
    first_beam = create_beam()
    second_beam = create_beam()

    positive_system = System()
    positive_system.add_input_beam(first_beam)
    positive_system.add(Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    ))

    negative_system = System()
    negative_system.add_input_beam(second_beam)
    negative_system.add(Interface(
        n1=1.0,
        n2=1.5,
        R=-0.1,
    ))

    positive = positive_system.build().run()
    negative = negative_system.build().run()

    assert not np.isclose(
        positive.final.q,
        negative.final.q,
    )


def test_curved_interface_propagates_new_medium_downstream():
    beam = create_beam()

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    )

    system = System()
    system.add_input_beam(beam)
    system.add(interface)
    system.add(Space(d=0.2))

    result = system.build().run()

    assert result.states[1].n == pytest.approx(1.5)
    assert result.states[2].n == pytest.approx(1.5)


def test_variable_curved_interface_radius_changes_result():
    beam = create_beam()

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    ).variable("R")

    system = System()
    system.add_input_beam(beam)
    system.add(interface)

    simulation = system.build()

    first = simulation.run(np.array([0.1]))
    second = simulation.run(np.array([0.2]))

    assert not np.isclose(
        first.final.q,
        second.final.q,
    )


def test_curved_interface_radius_is_differentiable():
    beam = create_beam()

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
    ).variable("R")

    system = System()
    system.add_input_beam(beam)
    system.add(interface)

    simulation = system.build()

    def objective(theta):
        return np.real(
            simulation.run(theta).final.q
        )

    derivative = grad(objective)(
        simulation.initial_values
    )

    assert derivative.shape == (1,)
    assert np.isfinite(derivative[0])
    assert not np.isclose(derivative[0], 0.0)



# -------------------
# CROSS-SYSTEM REUSE
# -------------------

def test_same_inherited_space_can_be_reused_across_systems():
    space = Space(d=0.2)

    air = System(ambient_n=1.0)
    air.add_input_beam(create_beam(n=1.0))
    air.add(space)

    glass = System(ambient_n=1.5)
    glass.add_input_beam(create_beam(n=1.5))
    glass.add(space)

    air_result = air.build().run()
    glass_result = glass.build().run()

    assert air_result.final.n == pytest.approx(1.0)
    assert glass_result.final.n == pytest.approx(1.5)


def test_reusing_element_in_second_system_does_not_change_first_simulation():
    space = Space(d=0.2)

    first_system = System(ambient_n=1.0)
    first_system.add_input_beam(create_beam(n=1.0))
    first_system.add(space)

    first_simulation = first_system.build()

    second_system = System(ambient_n=1.5)
    second_system.add_input_beam(create_beam(n=1.5))
    second_system.add(space)

    second_simulation = second_system.build()

    first = first_simulation.run()
    second = second_simulation.run()

    assert first.final.n == pytest.approx(1.0)
    assert second.final.n == pytest.approx(1.5)


def test_same_variable_element_can_be_run_with_different_theta_in_different_systems():
    space = Space(d=0.2).variable("d")

    first_system = System()
    first_system.add_input_beam(create_beam())
    first_system.add(space)

    second_system = System()
    second_system.add_input_beam(create_beam())
    second_system.add(space)

    first_simulation = first_system.build()
    second_simulation = second_system.build()

    first = first_simulation.run(np.array([0.3]))
    second = second_simulation.run(np.array([0.7]))

    assert first.z[-1] == pytest.approx(0.3)
    assert second.z[-1] == pytest.approx(0.7)

    assert first_simulation.initial_values[0] == pytest.approx(0.2)
    assert second_simulation.initial_values[0] == pytest.approx(0.2)


def test_same_variable_element_parameter_identity_is_preserved_across_simulations():
    space = Space(d=0.2).variable("d")
    parameter = space.d.node

    first_system = System()
    first_system.add_input_beam(create_beam())
    first_system.add(space)

    second_system = System()
    second_system.add_input_beam(create_beam())
    second_system.add(space)

    first_simulation = first_system.build()
    second_simulation = second_system.build()

    assert first_simulation.parameter_info[0].parameter is parameter
    assert second_simulation.parameter_info[0].parameter is parameter


def test_same_standalone_parameter_can_drive_different_systems():
    scale = Parameter(
        0.2,
        name="scale",
    ).variable()

    first_space = Space(d=scale)
    second_lens = ThinLens(f=scale)

    first_system = System()
    first_system.add_input_beam(create_beam())
    first_system.add(first_space)

    second_system = System()
    second_system.add_input_beam(create_beam())
    second_system.add(second_lens)

    first_simulation = first_system.build()
    second_simulation = second_system.build()

    assert len(first_simulation.initial_values) == 1
    assert len(second_simulation.initial_values) == 1

    assert first_simulation.parameter_info[0].parameter is scale
    assert second_simulation.parameter_info[0].parameter is scale

    first = first_simulation.run(np.array([0.4]))
    second = second_simulation.run(np.array([0.8]))

    assert first.z[-1] == pytest.approx(0.4)

    expected = second_simulation.source.q / (
        1.0 - second_simulation.source.q / 0.8
    )

    assert second.final.q == pytest.approx(expected)


def test_mutation_between_builds_only_affects_later_build():
    space = Space(d=0.2)

    first_system = System()
    first_system.add_input_beam(create_beam())
    first_system.add(space)

    first_simulation = first_system.build()

    space.d = 0.6

    second_system = System()
    second_system.add_input_beam(create_beam())
    second_system.add(space)

    second_simulation = second_system.build()

    first = first_simulation.run()
    second = second_simulation.run()

    assert first.z[-1] == pytest.approx(0.2)
    assert second.z[-1] == pytest.approx(0.6)


def test_same_composite_can_be_reused_across_systems():
    slab = Slab(
        d=0.02,
        n=1.5,
    )

    first_system = System(ambient_n=1.0)
    first_system.add_input_beam(create_beam(n=1.0))
    first_system.add(slab)

    second_system = System(ambient_n=1.0)
    second_system.add_input_beam(create_beam(n=1.0))
    second_system.add(slab)

    first = first_system.build().run()
    second = second_system.build().run()

    assert first.z[-1] == pytest.approx(0.02)
    assert second.z[-1] == pytest.approx(0.02)
    assert first.final.q == pytest.approx(second.final.q)


def test_same_composite_can_be_added_twice_to_one_system():
    beam = create_beam()

    slab = Slab(
        d=0.02,
        n=1.5,
    )

    system = System()
    system.add_input_beam(beam)
    system.add(slab)
    system.add(slab)

    simulation = system.build()
    result = simulation.run()

    assert len(simulation.steps) == 6
    assert len(result.states) == 7
    assert result.z[-1] == pytest.approx(0.04)


def test_repeated_composite_leaf_locations_are_disambiguated_by_occurrence():
    beam = create_beam()

    slab = Slab(
        d=0.02,
        n=1.5,
    )

    front, body, back = tuple(slab)

    system = System()
    system.add_input_beam(beam)
    system.add(slab)
    system.add(slab)

    result = system.build().run()

    assert result.at(front, occurrence=0) is result.states[0]
    assert result.after(front, occurrence=0) is result.states[1]
    assert result.at(back, occurrence=0) is result.states[2]
    assert result.after(back, occurrence=0) is result.states[3]

    assert result.at(front, occurrence=1) is result.states[3]
    assert result.after(front, occurrence=1) is result.states[4]
    assert result.at(back, occurrence=1) is result.states[5]
    assert result.after(back, occurrence=1) is result.states[6]


def test_variable_composite_reused_twice_keeps_single_design_dimension():
    beam = create_beam()

    slab = Slab(
        d=0.02,
        n=1.5,
    ).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(slab)
    system.add(slab)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.03]))

    assert result.z[-1] == pytest.approx(0.06)


def test_same_slab_resolves_ambient_independently_in_different_systems():
    slab = Slab(
        d=0.02,
        n=1.8,
    )

    air = System(ambient_n=1.0)
    air.add_input_beam(create_beam(n=1.0))
    air.add(slab)

    water = System(ambient_n=1.33)
    water.add_input_beam(create_beam(n=1.33))
    water.add(slab)

    air_result = air.build().run()
    water_result = water.build().run()

    assert air_result.initial.n == pytest.approx(1.0)
    assert air_result.final.n == pytest.approx(1.0)

    assert water_result.initial.n == pytest.approx(1.33)
    assert water_result.final.n == pytest.approx(1.33)