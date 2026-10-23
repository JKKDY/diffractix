"""System build pipelines, requirements, snapshots, and resolved metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import numpy as np
from diffractix.backend import xp

from diffractix.beams import GaussianBeam, Ray
from diffractix.composites import CompositeElement, Slab
from diffractix.elements import (
    ABCD,
    GaussianAperture,
    GRIN,
    Interface,
    Mirror,
    OpticalElement,
    Space,
    ThinLens,
)
from diffractix.graph import Node, Parameter, Symbol, compile_ast
from diffractix.solver import Solver, SolverContext
from diffractix.solver.constraint import Constraint
from diffractix.system import System, SystemValidationError
from diffractix.system.info import ElementInfo


def create_beam(n=1.0):
    return GaussianBeam.from_waist(
        w0=1e-3,
        wavelength=1064e-9,
        n=n,
    )


class NestedRequirementsComposite(CompositeElement):
    def __init__(self, inner, child):
        self.inner = inner
        self.child = child
        super().__init__()


def test_simulation_initial_values_are_read_only():
    distance = Parameter(0.2).variable()
    system = System()
    system.add_input_beam(create_beam())
    system.add(Space(d=distance))

    simulation = system.build()

    with pytest.raises(ValueError):
        simulation.initial_values[0] = 0.3


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
        xp.array([0.0, 0.1, 0.25, 0.25]),
    )
    assert result.at(lens) is result.states[2]
    assert result.after(lens) is result.states[3]


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


def test_system_requirements_are_forwarded_to_simulation():
    beam = create_beam()
    space = Space(d=0.1)
    requirement = space.d <= 0.2

    system = System()
    system.add_input_beam(beam)
    system.add(space)
    system.require(requirement)

    simulation = system.build()

    assert all(isinstance(requirement, Constraint) for requirement in simulation.requirements)
    assert simulation.requirements[0].evaluate is requirement.left
    assert simulation.requirements[1].evaluate is space.requirements[0].left


def test_element_requirement_is_forwarded_to_simulation():
    beam = create_beam()
    element = Space(d=0.1)
    length_requirement = element.requirements[0]
    requirement = element.d <= 0.2
    element.require(requirement)

    system = System()
    system.add_input_beam(beam)
    system.add(element)

    simulation = system.build()

    assert all(isinstance(requirement, Constraint) for requirement in simulation.requirements)
    assert simulation.requirements[0].evaluate is length_requirement.left
    assert simulation.requirements[1].evaluate is requirement.left


def test_system_and_nested_element_requirements_preserve_order():
    beam = create_beam()
    inner = Slab(d=0.01, n=1.5)
    child = Space(d=0.1)
    composite = NestedRequirementsComposite(inner, child)
    system_requirement = child.d <= 0.5
    composite_requirement = inner.d >= 0.005
    inner_requirement = inner.n >= 1.0
    child_requirement = child.d <= 0.2
    system = System()
    system.add_input_beam(beam)
    system.add(composite)
    system.require(system_requirement)
    composite.require(composite_requirement)
    inner.require(inner_requirement)
    child.require(child_requirement)

    simulation = system.build()

    assert all(isinstance(requirement, Constraint) for requirement in simulation.requirements)
    constraints = tuple(
        (requirement.lower_bound, requirement.upper_bound)
        for requirement in simulation.requirements
    )
    assert constraints[0] == (-math.inf, 0.5)
    assert constraints[1] == (0.005, math.inf)
    assert constraints[2] == (1.0, math.inf)
    assert constraints[3] == (math.nextafter(0.0, math.inf), math.inf)
    assert any(upper == pytest.approx(0.2) for _, upper in constraints)
    assert constraints[-3:] == ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def test_repeated_element_requirements_are_collected_once_without_mutation():
    beam = create_beam()
    element = Space(d=0.1)
    requirement = element.d <= 0.2
    element.require(requirement)
    original_requirements = element.requirements
    system = System()
    system.add_input_beam(beam)
    system.add((element, element))
    system_requirements = system.requirements

    simulation = system.build()

    assert len(simulation.requirements) == len(original_requirements)
    assert all(isinstance(requirement, Constraint) for requirement in simulation.requirements)
    assert element.requirements == original_requirements
    assert system.requirements == system_requirements


def test_builtin_element_requirements_reach_simulation_in_element_order():
    beam = create_beam()
    elements = (
        Space(d=0.01),
        GRIN(d=0.02, g=10.0, n=1.0),
        ABCD(thickness=0.03, n=1.0),
        GaussianAperture(a=1e-3),
    )
    system = System()
    system.add_input_beam(beam)
    system.add(elements)

    simulation = system.build()

    assert all(isinstance(requirement, Constraint) for requirement in simulation.requirements)
    assert [requirement.lower_bound for requirement in simulation.requirements] == [
        0.0,
        0.0,
        math.nextafter(0.0, math.inf),
        math.nextafter(0.0, math.inf),
        0.0,
        math.nextafter(0.0, math.inf),
        0.0,
    ]


@pytest.mark.parametrize("attach_to", ("system", "element"))
def test_malformed_requirements_fail_during_build(attach_to):
    element = Space(d=0.1)
    system = System()
    system.add_input_beam(create_beam())
    system.add(element)

    if attach_to == "system":
        system.require(object())
    else:
        element.require(object())

    with pytest.raises(TypeError):
        system.build()


def test_builtin_element_requirements_compile_for_solver():
    elements = (
        Space(d=0.01),
        GRIN(d=0.02, g=10.0, n=1.0),
        ABCD(thickness=0.03, n=1.0),
        GaussianAperture(a=1e-3),
    )
    system = System()
    system.add_input_beam(create_beam())
    system.add(elements)

    simulation = system.build()
    constraints = Solver(simulation)._compile_constraints(simulation.requirements)
    context = SimpleNamespace(theta=xp.array([]))

    assert [constraint.lower_bound for constraint in constraints] == [
        0.0,
        0.0,
        math.nextafter(0.0, math.inf),
        math.nextafter(0.0, math.inf),
        0.0,
        math.nextafter(0.0, math.inf),
        0.0,
    ]
    assert [constraint.upper_bound for constraint in constraints] == [
        math.inf,
        math.inf,
        math.inf,
        math.inf,
        math.inf,
        math.inf,
        0.0,
    ]
    assert [constraint.evaluate(context) for constraint in constraints] == pytest.approx(
        [0.01, 0.02, 1.0, 1.0, 0.03, 1e-3, 0.0],
    )


def test_grin_zero_gradient_builds_and_runs():
    system = System()
    system.add_input_beam(create_beam())
    system.add(GRIN(d=0.02, g=0.0, n=1.0))

    simulation = system.build()
    result = simulation.run(simulation.initial_values)

    assert result.z[-1] == pytest.approx(0.02)
    assert len(result.states) == 2


@pytest.mark.scipy
def test_interface_refractive_index_requirement_restricts_optimization():
    n1 = Parameter(1.0, variable=True, lower_bound=0.5, upper_bound=2.0)
    n2 = Parameter(1.5, variable=True, lower_bound=-2.0, upper_bound=2.0)
    system = System()
    system.ambient_n.variable(lower_bound=0.5, upper_bound=2.0)
    system.add_input_beam(create_beam())
    system.add(Space(d=0.01))
    interface = Interface(n1=n1, n2=n2)
    system.add(interface)
    simulation = system.build()

    assert len(simulation.requirements) == 4
    constraints = Solver(simulation)._compile_constraints(simulation.requirements)
    context = SimpleNamespace(theta=xp.array([1.0, 1.0, -1.0]))
    assert constraints[2].evaluate(context) == pytest.approx(-1.0)
    assert constraints[2].lower_bound > 0.0
    assert constraints[-1].lower_bound == constraints[-1].upper_bound == 0.0

    solver = Solver(simulation)
    solver.target(
        system.ambient_n - 1.5,
        n1 - 1.2,
        n2 + 0.5,
    )
    solution = solver.solve(method="SLSQP")

    assert solution[system.ambient_n] == pytest.approx(solution[n1], abs=1e-7)
    assert solution[system.ambient_n] > 1.0
    assert solution[n2] == pytest.approx(0.0, abs=1e-8)
    assert solution[n2] > -1e-8
    assert solution.constraints[-1].satisfied


@pytest.mark.parametrize(
    "element",
    (
        ThinLens(f=0.0),
        Interface(n1=1.0, n2=1.5, R=0.0),
        Mirror(R=0.0),
    ),
)
def test_system_build_rejects_singular_curvature_or_focal_length(element):
    system = System()
    system.add_input_beam(create_beam())
    system.add(element)

    with pytest.raises(SystemValidationError, match="must not be zero"):
        system.build()


def test_system_build_requires_input_beam():
    system = System()
    system.add(Space(d=0.1))

    with pytest.raises(SystemValidationError):
        system.build()


@dataclass(eq=False, kw_only=True)
class MalformedMatrixElement(OpticalElement):
    x: Node

    @property
    def matrix(self):
        return ((1.0, self.x),)

    @property
    def element_length(self):
        return 0.0


def make_beam():
    return Ray(
        x=0.0,
        theta=0.0,
    )


def test_variadic_system_matches_incremental_construction_and_run():
    import numpy as np

    beam = make_beam()
    elements = (Space(d=0.05), ThinLens(f=0.1), Space(d=0.1))
    concise = System(*elements, input=beam, ambient_n=1.33)
    incremental = System(ambient_n=1.33)
    incremental.add_input_beam(beam)
    incremental.add(elements)

    assert concise.beam is beam
    assert all(a is b for a, b in zip(concise.elements, elements))
    first, second = concise.build().run(), incremental.build().run()
    np.testing.assert_allclose(first.z, second.z)
    np.testing.assert_allclose(first.x, second.x)
    np.testing.assert_allclose(first.theta, second.theta)


def test_build_resolves_hashable_compile_context_key():
    system = System()
    key = ("lens", "focal_length")

    system.add_input_beam(make_beam())
    system.add_context(key, 0.2)
    system.add(ThinLens(f=Symbol(key)))

    simulation = system.build()

    assert simulation.parameter_graph.evaluate(
        simulation.parameter_graph.initial_values
    )[0] == pytest.approx(0.2)


def test_build_snapshots_compile_context():
    system = System()
    system.add_input_beam(make_beam())
    system.add_context("temperature", 293.15)
    system.add(Space(d=0.1))

    simulation = system.build()
    frozen_temperature = simulation.compile_context["temperature"]

    system.add_context("temperature", 300.0)

    assert simulation.compile_context["temperature"] is frozen_temperature
    assert frozen_temperature.value == pytest.approx(293.15)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_custom_numpy_scalar_matrix_compiles_and_propagates(dtype):
    @dataclass(eq=False, kw_only=True)
    class NumericMatrixElement(OpticalElement):
        @property
        def matrix(self):
            return np.array(((1.0, 0.02), (-2.0, 0.96)), dtype=dtype)

        @property
        def element_length(self):
            return np.float32(0.02)

    source = create_beam()
    element = NumericMatrixElement()
    simulation = System(element, input=source).build()
    result = simulation.run()
    A, B, C, D = element.matrix.ravel()

    assert result.final.q == pytest.approx((A * source.q + B) / (C * source.q + D))
    assert result.z[-1] == pytest.approx(0.02)


def test_build_reports_actual_element_graph_validation_error():
    system = System()
    system.add_input_beam(make_beam())
    system.add(MalformedMatrixElement(x=1.0, label="Malformed"))

    with pytest.raises(SystemValidationError) as exc_info:
        system.build()

    text = str(exc_info.value)
    assert "Malformed" in text
    assert "matrix must have exactly 2 rows" in text


class NestedComposite(CompositeElement):

    def __init__(self):
        self.lens = ThinLens(f=0.1, label="Lens")
        self.slab = Slab(d=0.01, n=1.5, label="Slab")
        super().__init__()


def build_simulation(*elements):
    system = System()
    system.add_input_beam(
        GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    )

    for element in elements:
        system.add(element)

    return system.build()


def test_build_still_rejects_initial_refractive_index_mismatch():
    system = System(ambient_n=1.0)
    system.add_input_beam(
        GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    )
    system.add(Space(d=0.1, n=1.5, label="Glass"))

    with pytest.raises(ValueError, match="Refractive index mismatch.*Glass"):
        system.build()


def test_build_appends_continuity_equality_constraint_after_existing_requirements():
    system = System(ambient_n=1.0)
    system.add_input_beam(
        GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    )
    refractive_index = Parameter(1.0, name="glass_n").variable()
    space = Space(d=0.1, n=refractive_index, label="Glass")
    existing_requirement = space.d <= 0.2
    space.require(existing_requirement)
    system.add(space)

    simulation = system.build()

    continuity = simulation.requirements[-1]
    assert continuity.evaluate is not existing_requirement
    assert continuity.lower_bound == 0.0
    assert continuity.upper_bound == 0.0
    assert continuity.label.startswith(
        "Refractive index continuity at Placement #0 (Space 'Glass')"
    )
    assert simulation.requirements[-2].upper_bound == 0.2
    assert len(simulation.requirements) == 4


def test_shared_refractive_index_parameter_remains_one_variable_with_continuity():
    system = System(ambient_n=1.0, ambient_n_variable=True)
    system.add_input_beam(
        GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    )
    system.add(Space(d=0.1, n=system.ambient_n))

    simulation = system.build()

    assert simulation.graph.variables == (system.ambient_n,)
    continuity = simulation.requirements[-1]
    graph = compile_ast(
        (continuity.evaluate,),
        parameter_snapshot=simulation.parameter_info,
    )
    assert graph.evaluate(xp.array([1.7]))[0] == pytest.approx(0.0)


def test_build_creates_element_info_for_each_resolved_step():
    space = Space(d=0.1, label="Drift")

    simulation = build_simulation(space)

    assert len(simulation.steps) == len(simulation.element_info) == 1
    assert simulation.element_info == (
        ElementInfo(
            type_name="Space",
            label="Drift",
            path=None,
            parameter_names=("d",),
            parameter_indices=(0,),
        ),
    )


def test_element_info_preserves_resolved_composite_paths():
    composite = NestedComposite()

    simulation = build_simulation(composite)

    assert tuple(info.type_name for info in simulation.element_info) == (
        "ThinLens",
        "Interface",
        "Space",
        "Interface",
    )
    assert tuple(info.label for info in simulation.element_info) == (
        "Lens",
        "Slab_In",
        "Slab_Body",
        "Slab_Out",
    )
    assert tuple(info.path for info in simulation.element_info) == (
        "lens",
        "slab.front",
        "slab.body",
        "slab.back",
    )


def test_parameter_graph_evaluates_fixed_and_variable_element_parameters():
    distance = Parameter(0.05, name="distance").variable()
    fixed_space = Space(d=0.2, label="Fixed")
    variable_space = Space(d=distance, label="Variable")

    simulation = build_simulation(fixed_space, variable_space)
    values = simulation.parameter_graph.evaluate(
        simulation.parameter_graph.initial_values
    )

    fixed_info, variable_info = simulation.element_info

    assert fixed_info.parameter_names == ("d",)
    assert values[fixed_info.parameter_indices[0]] == pytest.approx(0.2)
    assert variable_info.parameter_names == ("d",)
    assert values[variable_info.parameter_indices[0]] == pytest.approx(0.05)


def test_parameter_graph_evaluates_derived_parameters_and_skips_empty_inputs():
    distance = Parameter(0.05, name="distance").variable()
    space = Space(d=2 * distance, n=None)

    simulation = build_simulation(space)
    info = simulation.element_info[0]
    values = simulation.parameter_graph.evaluate(
        simulation.parameter_graph.initial_values
    )

    assert info.parameter_names == ("d",)
    assert info.parameter_indices == (0,)
    assert values[0] == pytest.approx(0.1)


def test_parameter_graph_preserves_canonical_simulation_metadata_and_snapshot():
    distance = Parameter(0.05, name="distance").variable()
    space = Space(d=2 * distance)

    simulation = build_simulation(space)
    initial_values = simulation.initial_values.copy()
    graph_variables = simulation.graph.variables
    parameter_index = simulation.parameter_info[id(distance)].parameter_index

    space.d = 0.4
    distance.value = 0.2

    parameter_values = simulation.parameter_graph.evaluate(
        simulation.parameter_graph.initial_values
    )

    assert simulation.initial_values == pytest.approx(initial_values)
    assert simulation.graph.variables == graph_variables == (distance,)
    assert simulation.parameter_info[id(distance)].parameter_index == parameter_index == 0
    assert parameter_values[0] == pytest.approx(0.1)


def test_shared_leaf_in_distinct_nested_composites_keeps_requirements_and_locations_separate():
    distance = Parameter(0.01).variable()
    shared = Space(d=distance, label="shared gap")
    left = NestedRequirementsComposite(shared, ThinLens(f=0.1))
    right = NestedRequirementsComposite(shared, ThinLens(f=0.2))
    outer = NestedRequirementsComposite(left, right)
    system = System(outer, input=create_beam())
    system.require(Constraint(distance, upper_bound=0.04, label="system"))
    outer.require(Constraint(distance, lower_bound=0.001, label="outer"))
    left.require(Constraint(distance, upper_bound=0.03, label="left"))
    shared.require(Constraint(distance, lower_bound=0.005, label="shared"))
    right.require(Constraint(distance, upper_bound=0.025, label="right"))
    owners = (system, outer, left, shared, right)
    original_requirements = [owner.requirements for owner in owners]

    simulation = system.build()

    # Shared ownership deduplicates requirements, not physical occurrences.
    assert [constraint.label for constraint in simulation.requirements] == ["system", "outer", "left", None, "shared", "right"]
    assert [info.path for info in simulation.element_info] == ["inner.inner", "inner.child", "child.inner", "child.child"]
    assert len(simulation.initial_values) == 1
    assert simulation.location_map[id(shared)] == ((0, 1), (2, 3))
    for owner, original in zip(owners, original_requirements):
        assert len(owner.requirements) == len(original)
        assert all(current is previous for current, previous in zip(owner.requirements, original))

    solver = Solver(simulation)
    solver.target(lambda ctx: ctx.z_after(shared, occurrence=1) - ctx.z_after(shared, occurrence=0))
    objective, = solver._compile_objectives(solver.objectives)
    assert objective.evaluate(SolverContext(xp.array([0.02]), simulation.run)) == pytest.approx(0.02)
