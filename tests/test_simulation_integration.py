from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
import autograd.numpy as np

from autograd import grad

from diffractix.beams import GaussianBeam
from diffractix.beams.base import ParaxialState
from diffractix.elements import Interface, Space, ThinLens
from diffractix.graph import Parameter
from diffractix.system.system import System
from diffractix.simulation import SimulationResult


# -------
# HELPERS
# -------

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


# ----------------
# EMPTY SIMULATION
# ----------------

def test_simulation_supports_empty_optical_path():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)

    simulation = system.build()
    result = simulation.run()

    assert simulation.steps == ()
    assert len(result.states) == 1
    assert result.initial is beam
    assert result.final is beam
    assert np.allclose(result.z, np.array([0.0]))


def test_empty_simulation_has_no_parameters():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)

    simulation = system.build()

    assert len(simulation.initial_values) == 0
    assert simulation.parameter_info == ()


# ----------------
# BASIC PROPAGATION
# ----------------

def test_simulation_propagates_single_space():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    result = system.build().run()

    assert len(result.states) == 2
    assert result.initial is beam
    assert result.z[-1] == pytest.approx(0.2)
    assert result.final.q == pytest.approx(beam.q + 0.2)


def test_simulation_propagates_single_thin_lens():
    beam = create_beam()
    lens = ThinLens(f=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)

    result = system.build().run()

    expected = beam.q / (1.0 - beam.q / 0.2)

    assert len(result.states) == 2
    assert np.allclose(result.z, np.array([0.0, 0.0]))
    assert result.final.q == pytest.approx(expected)


def test_simulation_propagates_mixed_sequence():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.1))
    system.add(ThinLens(f=0.2))
    system.add(Space(d=0.3))

    result = system.build().run()

    q1 = beam.q + 0.1
    q2 = q1 / (1.0 - q1 / 0.2)
    q3 = q2 + 0.3

    assert len(result.states) == 4
    assert np.allclose(
        result.z,
        np.array([0.0, 0.1, 0.1, 0.4]),
    )
    assert result.final.q == pytest.approx(q3)


def test_simulation_records_state_after_every_step():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.1))
    system.add(ThinLens(f=0.2))
    system.add(Space(d=0.3))

    simulation = system.build()
    result = simulation.run()

    assert len(result.states) == len(simulation.steps) + 1
    assert len(result.z) == len(result.states)


def test_simulation_positions_are_monotonic_non_decreasing():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.1))
    system.add(ThinLens(f=0.2))
    system.add(ThinLens(f=0.3))
    system.add(Space(d=0.4))

    result = system.build().run()

    assert np.all(np.diff(result.z) >= 0.0)


def test_zero_length_elements_preserve_duplicate_positions():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(ThinLens(f=0.2))
    system.add(ThinLens(f=0.3))

    result = system.build().run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.0, 0.0]),
    )


# -------------------
# DEFAULT PARAMETERS
# -------------------

def test_run_without_theta_matches_run_with_initial_values():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    default = simulation.run()
    explicit = simulation.run(simulation.initial_values)

    assert np.allclose(default.z, explicit.z)
    assert default.final.q == pytest.approx(explicit.final.q)


def test_run_uses_supplied_variable_parameter():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    result = simulation.run(np.array([0.5]))

    assert result.z[-1] == pytest.approx(0.5)
    assert result.final.q == pytest.approx(beam.q + 0.5)


def test_run_does_not_modify_supplied_theta():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    theta = np.array([0.5])
    original = theta.copy()

    simulation.run(theta)

    assert np.allclose(theta, original)


def test_run_rejects_wrong_theta_length():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    with pytest.raises(ValueError):
        simulation.run(np.array([]))


def test_fixed_parameters_do_not_appear_in_theta():
    beam = create_beam()
    space = Space(d=0.2)
    lens = ThinLens(f=0.3).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 1
    assert simulation.parameter_info[0].parameter is lens.f.node


# -----------------------
# MULTIPLE INDEPENDENT DOF
# -----------------------

def test_simulation_supports_multiple_independent_parameters():
    beam = create_beam()
    space = Space(d=0.2).variable("d")
    lens = ThinLens(f=0.3).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    simulation = system.build()

    theta = theta_with(
        simulation,
        (space.d.node, 0.4),
        (lens.f.node, 0.6),
    )

    result = simulation.run(theta)

    q1 = beam.q + 0.4
    expected = q1 / (1.0 - q1 / 0.6)

    assert len(simulation.initial_values) == 2
    assert result.z[-1] == pytest.approx(0.4)
    assert result.final.q == pytest.approx(expected)


def test_theta_parameter_mapping_does_not_depend_on_assumed_order():
    beam = create_beam()
    space = Space(d=0.2).variable("d")
    lens = ThinLens(f=0.3).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    simulation = system.build()

    theta = theta_with(
        simulation,
        (lens.f.node, 0.8),
        (space.d.node, 0.5),
    )

    result = simulation.run(theta)

    q1 = beam.q + 0.5
    expected = q1 / (1.0 - q1 / 0.8)

    assert result.final.q == pytest.approx(expected)


# -----------------
# SHARED PARAMETERS
# -----------------

def test_shared_parameter_controls_multiple_elements():
    beam = create_beam()
    distance = Parameter(0.1, name="distance").variable()

    first = Space(d=distance)
    second = Space(d=distance)

    system = System()
    system.add_input_beam(beam)
    system.add(first)
    system.add(second)

    simulation = system.build()

    assert len(simulation.initial_values) == 1
    assert simulation.parameter_info[0].parameter is distance

    result = simulation.run(np.array([0.3]))

    assert result.z[-1] == pytest.approx(0.6)
    assert result.final.q == pytest.approx(beam.q + 0.6)


def test_shared_parameter_between_different_element_types():
    beam = create_beam()
    scale = Parameter(0.2, name="scale").variable()

    space = Space(d=scale)
    lens = ThinLens(f=scale)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    simulation = system.build()
    result = simulation.run(np.array([0.4]))

    q1 = beam.q + 0.4
    expected = q1 / (1.0 - q1 / 0.4)

    assert len(simulation.initial_values) == 1
    assert result.final.q == pytest.approx(expected)


# ------------------
# DERIVED PARAMETERS
# ------------------

def test_derived_parameter_expression_is_evaluated_from_theta():
    beam = create_beam()
    scale = Parameter(0.1, name="scale").variable()
    space = Space(d=2 * scale)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()
    result = simulation.run(np.array([0.3]))

    assert len(simulation.initial_values) == 1
    assert result.z[-1] == pytest.approx(0.6)
    assert result.final.q == pytest.approx(beam.q + 0.6)


def test_shared_derived_parameter_expression_updates_entire_system():
    beam = create_beam()
    scale = Parameter(0.1, name="scale").variable()

    lens = ThinLens(f=2 * scale)
    space = Space(d=3 * scale)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(space)

    simulation = system.build()
    result = simulation.run(np.array([0.2]))

    q1 = beam.q / (1.0 - beam.q / 0.4)
    expected = q1 + 0.6

    assert len(simulation.initial_values) == 1
    assert result.z[-1] == pytest.approx(0.6)
    assert result.final.q == pytest.approx(expected)


# --------------------
# ABSOLUTE POSITIONING
# --------------------

def test_variable_absolute_position_controls_generated_space():
    beam = create_beam()
    position = Parameter(0.2, name="position").variable()
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

    assert result.at(lens) is result.states[1]
    assert result.after(lens) is result.states[2]


def test_variable_upstream_length_and_absolute_position_interact():
    beam = create_beam()

    space = Space(d=0.1).variable("d")
    position = Parameter(0.4, name="position").variable()
    lens = ThinLens(f=0.5)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens, z=position)

    simulation = system.build()

    theta = theta_with(
        simulation,
        (space.d.node, 0.2),
        (position, 0.6),
    )

    result = simulation.run(theta)

    assert np.allclose(
        result.z,
        np.array([0.0, 0.2, 0.6, 0.6]),
    )


# -----------------
# REPEATED ELEMENTS
# -----------------

def test_repeated_thin_lens_object_propagates_twice():
    beam = create_beam()
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    result = system.build().run()

    q1 = beam.q / (1.0 - beam.q / 0.4)
    q2 = q1 + 0.1
    expected = q2 / (1.0 - q2 / 0.4)

    assert result.final.q == pytest.approx(expected)


def test_repeated_element_occurrences_map_to_correct_states():
    beam = create_beam()
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    result = system.build().run()

    assert result.at(lens, occurrence=0) is result.states[0]
    assert result.after(lens, occurrence=0) is result.states[1]

    assert result.at(lens, occurrence=1) is result.states[2]
    assert result.after(lens, occurrence=1) is result.states[3]


def test_repeated_element_without_occurrence_is_ambiguous():
    beam = create_beam()
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    result = system.build().run()

    with pytest.raises(ValueError, match="occurs 2 times"):
        result.at(lens)

    with pytest.raises(ValueError, match="occurs 2 times"):
        result.after(lens)


def test_repeated_variable_element_has_one_theta_dimension():
    beam = create_beam()
    lens = ThinLens(f=0.4).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 1
    assert simulation.parameter_info[0].parameter is lens.f.node


def test_repeated_variable_element_uses_same_theta_value_each_time():
    beam = create_beam()
    lens = ThinLens(f=0.4).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    simulation = system.build()
    result = simulation.run(np.array([0.8]))

    q1 = beam.q / (1.0 - beam.q / 0.8)
    q2 = q1 + 0.1
    expected = q2 / (1.0 - q2 / 0.8)

    assert result.final.q == pytest.approx(expected)


def test_repeated_space_object_accumulates_length_twice():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(space)

    result = system.build().run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.2, 0.4]),
    )

    assert result.final.q == pytest.approx(beam.q + 0.4)


def test_repeated_variable_space_object_has_single_parameter():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(space)

    simulation = system.build()
    result = simulation.run(np.array([0.3]))

    assert len(simulation.initial_values) == 1
    assert result.z[-1] == pytest.approx(0.6)
    assert result.final.q == pytest.approx(beam.q + 0.6)


def test_consecutive_repeated_zero_length_element_occurrences():
    beam = create_beam()
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(lens)

    result = system.build().run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.0, 0.0]),
    )

    assert result.at(lens, occurrence=0) is result.states[0]
    assert result.after(lens, occurrence=0) is result.states[1]
    assert result.at(lens, occurrence=1) is result.states[1]
    assert result.after(lens, occurrence=1) is result.states[2]


def test_repeated_element_at_different_absolute_positions():
    beam = create_beam()
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(lens, z=0.1)
    system.add(lens, z=0.4)

    result = system.build().run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.1, 0.1, 0.4, 0.4]),
    )

    assert result.at(lens, occurrence=0) is result.states[1]
    assert result.after(lens, occurrence=0) is result.states[2]

    assert result.at(lens, occurrence=1) is result.states[3]
    assert result.after(lens, occurrence=1) is result.states[4]


# ----------------
# REFRACTIVE INDEX
# ----------------

def test_inherited_medium_is_propagated_into_state():
    beam = create_beam(n=1.5)
    space = Space(d=0.2)

    system = System(ambient_n=1.5)
    system.add_input_beam(beam)
    system.add(space)

    result = system.build().run()

    assert result.initial.n == pytest.approx(1.5)
    assert result.final.n == pytest.approx(1.5)


def test_interface_changes_state_refractive_index():
    beam = create_beam(n=1.0)

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=np.inf,
    )

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)

    result = system.build().run()

    assert result.initial.n == pytest.approx(1.0)
    assert result.final.n == pytest.approx(1.5)


def test_downstream_inherited_space_uses_interface_output_medium():
    beam = create_beam(n=1.0)

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=np.inf,
    )

    space = Space(d=0.2)

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)
    system.add(space)

    result = system.build().run()

    assert result.states[1].n == pytest.approx(1.5)
    assert result.states[2].n == pytest.approx(1.5)


def test_variable_interface_output_medium_propagates_downstream():
    beam = create_beam(n=1.0)

    n2 = Parameter(
        1.5,
        name="n2",
    ).variable()

    interface = Interface(
        n1=1.0,
        n2=n2,
        R=np.inf,
    )

    space = Space(d=0.2)

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)
    system.add(space)

    simulation = system.build()
    result = simulation.run(np.array([1.8]))

    assert len(simulation.initial_values) == 1
    assert result.states[1].n == pytest.approx(1.8)
    assert result.states[2].n == pytest.approx(1.8)


def test_interface_and_space_positions_are_recorded_correctly():
    beam = create_beam(n=1.0)

    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=np.inf,
    )

    system = System(ambient_n=1.0)
    system.add_input_beam(beam)
    system.add(interface)
    system.add(Space(d=0.2))

    result = system.build().run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.0, 0.2]),
    )


# --------------------
# COMPILED INDEPENDENCE
# --------------------

def test_simulation_is_independent_of_fixed_element_mutation_after_build():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    space.d = 0.8

    result = simulation.run()

    assert result.z[-1] == pytest.approx(0.2)
    assert result.final.q == pytest.approx(beam.q + 0.2)


def test_simulation_is_independent_of_variable_parameter_value_after_build():
    beam = create_beam()
    space = Space(d=0.2).variable("d")
    parameter = space.d.node

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    parameter.value = 0.9

    result = simulation.run()

    assert simulation.initial_values[0] == pytest.approx(0.2)
    assert result.z[-1] == pytest.approx(0.2)


def test_repeated_runs_do_not_leak_parameter_values():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    first = simulation.run(np.array([0.3]))
    second = simulation.run(np.array([0.7]))
    third = simulation.run(np.array([0.3]))

    assert first.z[-1] == pytest.approx(0.3)
    assert second.z[-1] == pytest.approx(0.7)
    assert third.z[-1] == pytest.approx(0.3)

    assert first.final.q == pytest.approx(third.final.q)


def test_run_does_not_mutate_source_beam():
    beam = create_beam()
    original_q = beam.q
    original_n = beam.n

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.5))

    simulation = system.build()
    simulation.run()

    assert beam.q == original_q
    assert beam.n == original_n


def test_result_initial_is_original_source_object():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.5))

    result = system.build().run()

    assert result.initial is beam
    assert result.source is beam


# -----------------
# DIFFERENTIABILITY
# -----------------

def test_simulation_is_differentiable_through_variable_space():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    def objective(theta):
        result = simulation.run(theta)
        return np.real(result.final.q)

    derivative = grad(objective)(simulation.initial_values)

    assert derivative.shape == (1,)
    assert derivative[0] == pytest.approx(1.0)


def test_simulation_is_differentiable_through_derived_expression():
    beam = create_beam()
    scale = Parameter(0.2, name="scale").variable()
    space = Space(d=3 * scale)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    def objective(theta):
        return np.real(simulation.run(theta).final.q)

    derivative = grad(objective)(simulation.initial_values)

    assert derivative[0] == pytest.approx(3.0)


def test_simulation_is_differentiable_through_repeated_element():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(space)

    simulation = system.build()

    def objective(theta):
        return np.real(simulation.run(theta).final.q)

    derivative = grad(objective)(simulation.initial_values)

    assert derivative[0] == pytest.approx(2.0)


def test_simulation_gradient_is_repeatable():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    def objective(theta):
        return np.real(simulation.run(theta).final.q)

    gradient = grad(objective)

    first = gradient(np.array([0.2]))
    second = gradient(np.array([0.8]))

    assert first[0] == pytest.approx(1.0)
    assert second[0] == pytest.approx(1.0)


# -----------------
# RESULT INTEGRATION
# -----------------

def test_unique_element_location_lookup():
    beam = create_beam()
    space = Space(d=0.2)
    lens = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens)

    result = system.build().run()

    assert result.at(space) is result.states[0]
    assert result.after(space) is result.states[1]

    assert result.at(lens) is result.states[1]
    assert result.after(lens) is result.states[2]


def test_arbitrary_numeric_probe_is_currently_unsupported():
    beam = create_beam()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=1.0))

    result = system.build().run()

    with pytest.raises(
        ValueError,
        match="does not support arbitrary-z probing",
    ):
        result.at(0.5)


def test_location_lookup_rejects_element_not_in_simulation():
    beam = create_beam()
    included = ThinLens(f=0.4)
    missing = ThinLens(f=0.4)

    system = System()
    system.add_input_beam(beam)
    system.add(included)

    result = system.build().run()

    with pytest.raises(KeyError, match="not part of this simulation"):
        result.at(missing)


# --------------------------
# GENERIC PARAXIAL STATE API
# --------------------------

class AffineResult:

    @property
    def values(self):
        return np.stack([
            state.value
            for state in self.states
        ])


@dataclass(frozen=True)
class AffineState(ParaxialState):
    result_type: ClassVar[type] = AffineResult

    value: float

    def propagate(self, A, B, C, D, n):
        return AffineState(
            value=A * self.value + B,
        )


def test_simulation_supports_non_gaussian_paraxial_state():
    source = AffineState(value=2.0)
    system = System()

    system.add_input_beam(source)
    system.add(Space(d=0.5))
    
    result = system.build().run()
    assert isinstance(result, SimulationResult)
    assert np.allclose(result.value, np.array([2.0, 2.5]))



def test_generated_result_property_can_read_full_simulation_trace():
    source = AffineState(value=2.0)
    system = System()

    system.add_input_beam(source)
    system.add(Space(d=0.5))
    system.add(Space(d=1.0))
    
    result = system.build().run()
    assert np.allclose(
        result.value,
        np.array([2.0, 2.5, 3.5]),
    )