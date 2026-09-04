from __future__ import annotations

import pytest
import autograd.numpy as np

from diffractix.beams import GaussianBeam
from diffractix.elements import Interface, Space, ThinLens
from diffractix.graph import Parameter
from diffractix.system import System, SystemValidationError


# -------
# HELPERS
# -------

def create_beam(n=1.0):
    return GaussianBeam.from_waist(
        w0=1e-3,
        wavelength=1064e-9,
        n=n,
    )


# -----------
# BASIC BUILD
# -----------

def test_system_builds_and_runs_single_space():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()
    result = simulation.run()

    assert len(result.states) == 2
    assert np.allclose(result.z, np.array([0.0, 0.2]))
    assert np.isclose(result.final.q, beam.q + 0.2)


def test_system_builds_and_runs_single_thin_lens():
    beam = create_beam()
    lens = ThinLens(f=0.1)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)

    result = system.build().run()

    expected_q = beam.q / (1.0 - beam.q / 0.1)

    assert len(result.states) == 2
    assert np.allclose(result.z, np.array([0.0, 0.0]))
    assert np.isclose(result.final.q, expected_q)


def test_system_propagates_multiple_elements_in_order():
    beam = create_beam()
    first_space = Space(d=0.1)
    lens = ThinLens(f=0.2)
    second_space = Space(d=0.3)

    system = System()
    system.add_input_beam(beam)
    system.add(first_space)
    system.add(lens)
    system.add(second_space)

    result = system.build().run()

    q1 = beam.q + 0.1
    q2 = q1 / (1.0 - q1 / 0.2)
    q3 = q2 + 0.3

    assert len(result.states) == 4
    assert np.allclose(
        result.z,
        np.array([0.0, 0.1, 0.1, 0.4]),
    )
    assert np.isclose(result.final.q, q3)


def test_system_accepts_element_sequence():
    beam = create_beam()
    first_space = Space(d=0.1)
    lens = ThinLens(f=0.2)
    second_space = Space(d=0.3)

    system = System()
    system.add_input_beam(beam)
    system.add((
        first_space,
        lens,
        second_space,
    ))

    result = system.build().run()

    assert len(result.states) == 4
    assert np.allclose(
        result.z,
        np.array([0.0, 0.1, 0.1, 0.4]),
    )


# ------
# LAYOUT
# ------

def test_system_resolves_absolute_element_position():
    beam = create_beam()
    space = Space(d=0.1)
    lens = ThinLens(f=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens, z=0.25)

    result = system.build().run()

    assert len(result.states) == 4
    assert np.allclose(
        result.z,
        np.array([0.0, 0.1, 0.25, 0.25]),
    )


def test_system_location_lookup_matches_resolved_element_position():
    beam = create_beam()
    space = Space(d=0.1)
    lens = ThinLens(f=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.add(lens, z=0.25)

    result = system.build().run()

    assert result.at(lens) is result.states[2]
    assert result.after(lens) is result.states[3]


# ----------------
# REFRACTIVE INDEX
# ----------------

def test_system_space_inherits_ambient_refractive_index():
    beam = create_beam(n=1.5)
    space = Space(d=0.2)

    system = System(ambient_n=1.5)
    system.add_input_beam(beam)
    system.add(space)

    result = system.build().run()

    assert result.final.n == pytest.approx(1.5)


def test_system_interface_changes_propagation_medium():
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


def test_system_propagates_in_new_medium_after_interface():
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


# -------------------
# VARIABLE PARAMETERS
# -------------------

def test_system_variable_parameter_controls_simulation():
    beam = create_beam()
    space = Space(d=0.2).variable("d")

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    initial = simulation.run()
    changed = simulation.run(np.array([0.4]))

    assert len(simulation.initial_values) == 1
    assert simulation.initial_values[0] == pytest.approx(0.2)

    assert initial.final.q == pytest.approx(beam.q + 0.2)
    assert changed.final.q == pytest.approx(beam.q + 0.4)

    assert initial.z[-1] == pytest.approx(0.2)
    assert changed.z[-1] == pytest.approx(0.4)


def test_system_shared_parameter_is_single_simulation_dimension():
    beam = create_beam()
    focal_length = Parameter(
        0.2,
        name="shared_f",
    ).variable()

    first_lens = ThinLens(f=focal_length)
    second_lens = ThinLens(f=focal_length)

    system = System()
    system.add_input_beam(beam)
    system.add(first_lens)
    system.add(Space(d=0.1))
    system.add(second_lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 1
    assert len(simulation.parameter_info) == 1
    assert simulation.parameter_info[0].parameter is focal_length


# ---------------
# BUILD ISOLATION
# ---------------

def test_built_simulation_is_independent_of_later_element_changes():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    simulation = system.build()

    space.d = 0.5

    result = simulation.run()

    assert result.z[-1] == pytest.approx(0.2)
    assert result.final.q == pytest.approx(beam.q + 0.2)


def test_separate_builds_snapshot_different_element_values():
    beam = create_beam()
    space = Space(d=0.2)

    system = System()
    system.add_input_beam(beam)
    system.add(space)

    first_simulation = system.build()

    space.d = 0.5

    second_simulation = system.build()

    first_result = first_simulation.run()
    second_result = second_simulation.run()

    assert first_result.z[-1] == pytest.approx(0.2)
    assert second_result.z[-1] == pytest.approx(0.5)

    assert first_result.final.q == pytest.approx(beam.q + 0.2)
    assert second_result.final.q == pytest.approx(beam.q + 0.5)


# -----------------
# REPEATED ELEMENTS
# -----------------

def test_system_supports_repeated_element_object():
    beam = create_beam()
    lens = ThinLens(f=0.2)
    space = Space(d=0.1)

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(space)
    system.add(lens)

    result = system.build().run()

    assert len(result.states) == 4

    assert result.at(lens, occurrence=0) is result.states[0]
    assert result.after(lens, occurrence=0) is result.states[1]

    assert result.at(lens, occurrence=1) is result.states[2]
    assert result.after(lens, occurrence=1) is result.states[3]


def test_system_repeated_element_requires_explicit_occurrence():
    beam = create_beam()
    lens = ThinLens(f=0.2)

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


def test_system_repeated_element_uses_same_parameter_values():
    beam = create_beam()
    lens = ThinLens(f=0.2).variable("f")

    system = System()
    system.add_input_beam(beam)
    system.add(lens)
    system.add(Space(d=0.1))
    system.add(lens)

    simulation = system.build()

    assert len(simulation.initial_values) == 1

    result = simulation.run(np.array([0.4]))

    q1 = beam.q / (1.0 - beam.q / 0.4)
    q2 = q1 + 0.1
    expected = q2 / (1.0 - q2 / 0.4)

    assert result.final.q == pytest.approx(expected)


# ------------
# REQUIREMENTS
# ------------

def test_system_requirements_are_forwarded_to_simulation():
    beam = create_beam()
    requirement = object()

    system = System()
    system.add_input_beam(beam)
    system.add(Space(d=0.1))
    system.require(requirement)

    simulation = system.build()

    assert simulation.requirements == (requirement,)


# ----------
# VALIDATION
# ----------

def test_system_build_requires_input_beam():
    system = System()
    system.add(Space(d=0.1))

    with pytest.raises(SystemValidationError):
        system.build()