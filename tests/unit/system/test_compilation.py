"""System build passes with real graph programs and resolved metadata."""

import numpy as np
import pytest

from diffractix import GaussianBeam, Ray, RayBundle
from diffractix.composites import CompositeElement, Slab
from diffractix.elements import GaussianAperture, Interface, Plane, Space, ThinLens
from diffractix.graph import Literal, Parameter, Symbol
from diffractix.solver.constraint import Constraint
from diffractix.system import System
from diffractix.system.info import ParameterData, ResolvedPlacement


class NestedComposite(CompositeElement):
    def __init__(self):
        self.lens = ThinLens(f=0.1, label="Focus")
        self.slab = Slab(d=0.01, n=1.5)
        super().__init__()


@pytest.mark.parametrize("source", [
    Ray(x=0.01, theta=0.02, n=None),
    GaussianBeam(q=1j, wavelength=1e-6, n=None),
])
@pytest.mark.parametrize("with_element", [False, True])
def test_input_without_index_uses_ambient_in_numerical_source(source, with_element):
    ambient = Parameter(1.33).variable()
    system = System(optical_input=source, ambient_n=ambient)
    if with_element:
        system.add(Space(d=0.1))
    simulation = system.build()

    assert source.n.node is None
    assert simulation.run().source.n == pytest.approx(1.33)
    guess = simulation.initial_values
    guess[ambient] = 1.5
    result = simulation.run(guess.values)
    assert result.source.n == pytest.approx(1.5)
    assert result.final.n == pytest.approx(1.5)
    assert source.n.node is None


def test_inherited_input_index_remains_differentiable():
    from diffractix.backend import grad, jit, xp

    ambient = Parameter(1.33).variable()
    simulation = System(optical_input=Ray(x=0, theta=0, n=None), ambient_n=ambient).build()
    index = lambda theta: simulation.run(theta).source.n
    theta = xp.array([1.5])
    np.testing.assert_allclose(grad(index)(theta), [1.0])
    np.testing.assert_allclose(jit(grad(index))(theta), [1.0])


def compile_optics(system, *, extra_continuity=()):
    """Use real resolution passes without constructing a Result or Solver."""
    bindings = system._collect_bindings()
    placements, metadata = system._resolve_layout(system._resolve_elements(), bindings)
    resolved, continuity = system._resolve_refractive_indices(placements, bindings)
    products = system._compile_optics(
        placements=resolved, optical_inputs=system._inputs, bindings=bindings,
        continuity_constraints=(*continuity, *extra_continuity), element_data=metadata,
    )
    return (*products, continuity)


def test_flattening_preserves_paths_identity_call_site_and_first_absolute_position():
    relay = NestedComposite()
    system = System().add(relay, z=0.5)
    original = system.placements[0]
    resolved = system._resolve_elements()
    assert [p.path for p in resolved] == ["lens", "slab.front", "slab.body", "slab.back"]
    assert all(p.element is leaf for p, leaf in zip(resolved, relay.leaf_elements))
    assert [p.z for p in resolved] == [0.5, None, None, None]
    assert all(p.call_site is original.call_site for p in resolved)
    assert system.placements == (original,) and original.element is relay and original.z == 0.5
    assert relay.element_names == ("lens", "slab")


def test_relative_placement_identity_survives_flattening():
    system = System(ThinLens(f=0.1))
    assert system._resolve_elements()[0] is system.placements[0]


def test_layout_accounts_for_lengths_and_multiple_absolute_positions():
    first, middle, last = Space(d=0.2), Space(d=0.1), ThinLens(f=0.1)
    system = System(first).add(middle, z=0.5).add(last, z=1.0)
    original = system.placements
    resolved, metadata = system._resolve_layout(system._resolve_elements(), system.bindings)
    assert [type(p.element).__name__ for p in resolved] == ["Space", "Space", "Space", "Space", "ThinLens"]
    assert resolved[0].element is first and resolved[2].element is middle and resolved[4].element is last
    assert resolved[1].element.d.evaluate() == pytest.approx(0.3)
    assert resolved[3].element.d.evaluate() == pytest.approx(0.4)
    assert all(p.z is None for p in resolved)
    assert [data.position for data in metadata] == list(range(5))
    assert system.placements == original and original[1].z == 0.5


def test_symbolic_gap_tracks_absolute_position_and_upstream_length():
    distance, target = Parameter(0.2).variable(), Parameter(0.5).variable()
    system = System(Space(d=distance)).add(Plane(label="Target"), z=target)
    resolved, metadata = system._resolve_layout(system._resolve_elements(), system.bindings)
    gap = resolved[1].element
    assert gap.d.evaluate() == pytest.approx(0.3)
    assert metadata[1].variable_parameter_ids == frozenset((id(distance), id(target)))
    distance.value, target.value = 0.3, 0.7
    assert gap.d.evaluate() == pytest.approx(0.4) and gap.label == "AutoSpace_to_Target"


def test_layout_preserves_paths_and_call_site_on_generated_space():
    slab = Slab(d=0.01, n=1.5)
    system = System().add(slab, z=0.1)
    original = system.placements[0]
    resolved, metadata = system._resolve_layout(system._resolve_elements(), system.bindings)
    assert [p.path for p in resolved] == ["front", "front", "body", "back"]
    assert all(p.call_site is original.call_site for p in resolved)
    assert [info.path for info in metadata] == ["front", "front", "body", "back"]
    assert all(info.element_id == id(p.element) for info, p in zip(metadata, resolved))


@pytest.mark.parametrize("source_n, ambient_n", [(1.5, 1.0), (1.0, 1.33)])
def test_source_medium_wins_and_inherited_element_field_is_not_assigned(source_n, ambient_n):
    source = Ray(x=0, theta=0, n=source_n)
    space = Space(d=0.1)
    system = System(space, optical_input=source, ambient_n=ambient_n)
    placement = system.placements[0]
    resolved, continuity = system._resolve_refractive_indices((placement,), system.bindings)
    assert isinstance(resolved[0], ResolvedPlacement) and resolved[0].placement is placement
    assert resolved[0].refractive_index.evaluate(system.bindings) == source_n
    assert continuity == () and space.n.node is None


@pytest.mark.parametrize("source", [Ray(x=0, theta=0, n=None), Ray(x=0, theta=0)])
def test_absent_or_ambient_symbol_source_index_resolves_to_ambient(source):
    system = System(Plane(), optical_input=source, ambient_n=1.33)
    resolved, continuity = system._resolve_refractive_indices(system.placements, system.bindings)
    assert resolved[0].refractive_index.evaluate(system.bindings) == 1.33 and continuity == ()


def test_interface_changes_medium_and_downstream_elements_inherit_it():
    source = Ray(x=0, theta=0, n=1.0)
    front, body, back = Interface(n1=1.0, n2=1.5), Space(d=0.01, n=1.5), Interface(n1=1.5, n2=1.0)
    system = System(front, body, back, Plane(), optical_input=source)
    resolved, continuity = system._resolve_refractive_indices(system.placements, system.bindings)
    assert [p.refractive_index.evaluate(system.bindings) for p in resolved] == [1.5, 1.5, 1.0, 1.0]
    assert continuity == ()
    assert all(p.element is declared for p, declared in zip(resolved, system.elements))


@pytest.mark.parametrize("element", [Space(d=0.1, n=1.5), Interface(n1=1.5, n2=1.6)])
def test_fixed_medium_mismatch_fails_during_resolution_with_location(element):
    system = System(element, optical_input=Ray(x=0, theta=0, n=1.0))
    with pytest.raises(ValueError, match="Refractive index mismatch at Placement"):
        system._resolve_refractive_indices(system.placements, system.bindings)


def test_variable_continuity_is_an_equality_even_when_initially_unsatisfied():
    source_n, required_n = Parameter(1.0).variable(), Parameter(1.5).variable()
    space = Space(d=0.1, n=required_n, label="Glass")
    system = System(space, optical_input=Ray(x=0, theta=0, n=source_n))
    resolved, continuity = system._resolve_refractive_indices(system.placements, system.bindings)
    assert len(continuity) == 1
    constraint = continuity[0]
    assert constraint.lower_bound == constraint.upper_bound == 0.0
    assert "Refractive index continuity at" in constraint.label and "Glass" in constraint.label
    assert constraint.evaluate.evaluate() == -0.5
    source_n.value = 1.5
    assert constraint.evaluate.evaluate() == 0.0
    assert resolved[0].refractive_index is required_n


def test_continuity_binding_reaches_variable_parameter_through_symbol():
    index = Parameter(1.5).variable()
    element = Space(d=0.1, n=Symbol("glass"))
    system = System(element, optical_input=Ray(x=0, theta=0, n=1.5)).bind("glass", index)
    resolved, continuity = system._resolve_refractive_indices(system.placements, system.bindings)
    assert len(continuity) == 1 and continuity[0].evaluate.evaluate(system.bindings) == 0.0
    assert resolved[0].refractive_index is element.n.node


def test_optics_deduplicates_shared_parameters_and_preserves_canonical_order():
    distance = Parameter(0.1, name="distance").variable(bounds=(0.01, 0.2))
    focal = Parameter(0.2, name="focal").variable()
    unused = Parameter(99.0).variable()
    source = Ray(x=distance * 0.01, theta=0.01, n=1.0)
    system = System(Space(d=distance), ThinLens(f=focal), ThinLens(f=focal), optical_input=source)
    system.bind("unused", unused)
    program, stages, inputs, parameters, metadata, _ = compile_optics(system)
    assert program.parameter_indices == {id(distance): 0, id(focal): 1}
    assert [info.parameter_id for info in parameters] == [id(distance), id(focal)]
    assert [info.parameter_index for info in parameters] == [0, 1]
    assert all(isinstance(info, ParameterData) for info in parameters)
    assert parameters[0].value == 0.1
    assert (parameters[0].lower_bound, parameters[0].upper_bound) == (0.01, 0.2)
    values = program.evaluate([0.15, 0.3])
    assert values[stages[0].length_index] == 0.15
    assert values[stages[1].matrix_indices[1][0]] == pytest.approx(-1 / 0.3)
    assert inputs[0].construct(values).x == pytest.approx(0.0015)
    assert len(metadata) == len(stages) == 3


def test_fixed_parameters_stay_outside_theta_but_have_inspection_outputs():
    distance = Parameter(0.05).variable()
    space, lens = Space(d=2 * distance), ThinLens(f=0.2, label="Focus")
    system = System(space, lens, optical_input=Ray(x=0, theta=0, n=1.0))
    program, stages, _, parameters, metadata, _ = compile_optics(system)
    values = program.evaluate([0.08])
    assert [info.parameter_id for info in parameters] == [id(distance)]
    assert metadata[0].parameter_names == ("d",) and metadata[0].variable_parameter_ids == frozenset((id(distance),))
    assert metadata[1].parameter_names == ("f",) and metadata[1].variable_parameter_ids == frozenset()
    assert values[metadata[0].parameter_indices[0]] == pytest.approx(0.16)
    assert values[metadata[1].parameter_indices[0]] == 0.2
    assert values[stages[0].length_index] == pytest.approx(0.16)
    assert metadata[1].label == "Focus" and metadata[1].type_name == "ThinLens"


def test_continuity_only_dependencies_extend_theta_after_optical_roots():
    distance = Parameter(0.1).variable()
    additional = Parameter(1.5, name="continuity_only").variable()
    continuity = Constraint(additional - 1.5, lower_bound=0, upper_bound=0)
    system = System(Space(d=distance), optical_input=Ray(x=0, theta=0, n=1.0))
    program, _, _, parameters, _, _ = compile_optics(system, extra_continuity=(continuity,))
    assert program.parameter_indices == {id(distance): 0, id(additional): 1}
    assert [info.parameter_id for info in parameters] == [id(distance), id(additional)]
    assert program.evaluate([0.2, 1.6])[-1] == 1.6


def test_compilation_snapshots_values_membership_handles_and_binding_expressions():
    fixed = Parameter(0.1)
    variable = Parameter(0.2).variable(bounds=(0.05, 0.5))
    lens = ThinLens(f=Symbol("focal"))
    system = System(Space(d=fixed), lens, optical_input=Ray(x=0, theta=0, n=1.0))
    expression = 2 * variable
    system.bind("focal", expression)
    program, stages, _, parameters, metadata, _ = compile_optics(system)
    fixed.value = 9.0
    variable.fixed().bound(0.1, 0.3)
    expression.right = Parameter(9.0)
    lens.f = 8.0
    system.bind("focal", 10.0)
    values = program.evaluate([0.25])
    assert values[stages[0].length_index] == 0.1
    assert values[stages[1].matrix_indices[1][0]] == -2.0
    assert values[metadata[1].parameter_indices[0]] == 0.5
    assert parameters[0].value == 0.2
    assert (parameters[0].lower_bound, parameters[0].upper_bound) == (0.05, 0.5)
    assert program.parameter_indices == {id(variable): 0}


def test_source_snapshot_owns_bundle_arrays_and_reconstructs_graph_fields():
    positions, angles = np.array([-0.001, 0.002]), np.array([0.01, -0.02])
    index = Parameter(1.5).variable()
    source = RayBundle(x=positions, theta=angles, n=index)
    system = System(Plane(), optical_input=source)
    program, _, inputs, parameters, _, _ = compile_optics(system)
    assert inputs[0].initial_state is not source
    assert not np.shares_memory(inputs[0].initial_state.x, positions)
    positions[:] = 99
    angles[:] = -99
    source.n = 2.0
    numerical = inputs[0].construct(program.evaluate([1.6]))
    np.testing.assert_array_equal(numerical.x, [-0.001, 0.002])
    np.testing.assert_array_equal(numerical.theta, [0.01, -0.02])
    assert numerical.n == 1.6 and parameters[0].value == 1.5 and numerical.parameter_names == ()


def test_complex_aperture_roots_preserve_real_source_wavelength():
    source = GaussianBeam(q=0.02j, wavelength=1.064e-6, n=1.5)
    system = System(GaussianAperture(a=0.001), optical_input=source)
    program, stages, inputs, _, _, _ = compile_optics(system)
    values = program.evaluate([])
    expected_c = -1j * 1.064e-6 / (np.pi * 1.5 * 0.001**2)
    np.testing.assert_allclose(values[stages[0].matrix_indices[1][0]], expected_c)
    numerical = inputs[0].construct(values)
    assert not np.iscomplexobj(numerical.wavelength) and numerical.wavelength == 1.064e-6


def test_unresolved_optical_symbol_is_rejected_even_with_interpreter_default():
    system = System(ThinLens(f=Symbol("missing_focal", default=0.1)),
                    optical_input=Ray(x=0, theta=0, n=1.0))
    with pytest.raises(ValueError, match="Unresolved optical Symbol.*missing_focal"):
        compile_optics(system)


def test_repeated_elements_keep_separate_steps_and_metadata_but_one_theta_slot():
    lens = ThinLens(f=0.1).variable("f")
    system = System(lens, Space(d=0.2), lens, optical_input=Ray(x=0, theta=0, n=1.0))
    program, stages, _, parameters, metadata, _ = compile_optics(system)
    assert len(stages) == len(metadata) == 3 and len(parameters) == 1
    assert [data.element_id for data in metadata] == [id(lens), id(system.elements[1]), id(lens)]
    values = program.evaluate([0.2])
    assert values[stages[0].matrix_indices[1][0]] == values[stages[2].matrix_indices[1][0]] == -5


@pytest.mark.parametrize("constant, lower, upper", [(2.0, 1.0, 3.0), (2.0, 2.0, 2.0), (0.0, 0.0, np.inf)])
def test_satisfied_fixed_constraints_are_removed_including_exact_boundaries(constant, lower, upper):
    expression = Parameter(constant) * 2 - constant
    constraint = Constraint(expression, lower, upper, label="constant")
    program, descriptors, callables = System()._compile_constraints((constraint,), {}, {})
    assert program.evaluate([]) == () and descriptors == () and callables == ()
    assert constraint.evaluate is expression and expression.left.left.value == constant


@pytest.mark.parametrize("value", [-1.0, 3.0])
def test_violated_fixed_constraint_is_reported_as_unsatisfiable(value):
    constraint = Constraint(Parameter(value), 0.0, 2.0, label="fixed clearance")
    with pytest.raises(ValueError, match="fixed clearance.*unsatisfiable constant constraint"):
        System()._compile_constraints((constraint,), {}, {})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_fixed_constraint_fails_preparation(value):
    with pytest.raises(ValueError, match="finite check.*non-finite constant"):
        System()._compile_constraints((Constraint(Literal(value), label="finite check"),), {}, {})


def test_variable_constraint_can_start_unsatisfied_and_retains_bounds():
    variable = Parameter(-1.0).variable()
    constraint = Constraint(variable, 0.0, 2.0, label="clearance")
    program, descriptors, callables = System()._compile_constraints((constraint,), {}, {id(variable): 0})
    assert program.evaluate([-1.0]) == (-1.0,) and program.evaluate([1.0]) == (1.0,)
    assert descriptors[0].lower_bound == 0.0 and descriptors[0].upper_bound == 2.0
    assert descriptors[0].label == "clearance" and callables == ()


@pytest.mark.parametrize("through_symbol", [False, True])
def test_constraints_cannot_introduce_variables_outside_model_theta(through_symbol):
    external = Parameter(1.0, name="external").variable()
    expression = Symbol("external") if through_symbol else external + 1
    bindings = {"external": 2 * external} if through_symbol else {}
    with pytest.raises(ValueError, match="outside the optical model"):
        System()._compile_constraints((expression >= 0,), bindings, {})


@pytest.mark.parametrize("default", [None, 1.0])
def test_constraint_symbols_need_explicit_compile_bindings(default):
    with pytest.raises(ValueError, match="unresolved Symbol binding.*missing"):
        System()._compile_constraints((Symbol("missing", default=default) >= 0,), {}, {})


def test_bound_symbol_resolves_and_fixed_external_parameter_is_allowed():
    variable = Parameter(2.0).variable()
    bindings = {"gain": variable * 2, "offset": Parameter(1.0)}
    program, descriptors, _ = System()._compile_constraints(
        (Symbol("gain") - Symbol("offset") >= 0,), bindings, {id(variable): 0},
    )
    assert program.evaluate([3.0]) == (5.0,) and descriptors[0].lower_bound == 0


def test_graph_and_callable_positions_preserve_interleaving_without_tracing():
    variable = Parameter(2.0).variable()
    calls = []

    def dynamic(context):
        calls.append(context)
        return variable

    deferred = Constraint(dynamic, lower_bound=1, label="dynamic")
    first = Constraint(variable * 2, upper_bound=5, label="first graph")
    last = Constraint(variable - 1, lower_bound=0, label="last graph")
    program, descriptors, callables = System()._compile_constraints((first, deferred, last), {}, {id(variable): 0})
    assert calls == []
    assert [data.position for data in descriptors] == [0, 2]
    assert [data.label for data in descriptors] == ["first graph", "last graph"]
    assert callables[0][0] == 1 and callables[0][1] is deferred
    assert program.evaluate([2.0]) == (4.0, 1.0)


def test_constraint_program_local_order_does_not_change_model_order():
    first, second = Parameter(1.0).variable(), Parameter(2.0).variable()
    canonical = {id(first): 0, id(second): 1}
    program, _, _ = System()._compile_constraints((second - 2 * first >= 0,), {}, canonical)
    assert canonical == {id(first): 0, id(second): 1}
    assert program.parameter_indices == {id(second): 0, id(first): 1}
    assert program.evaluate([7.0, 3.0]) == (1.0,)


def test_zero_argument_constraint_is_normalized_once_and_graph_snapshotted():
    parameter = Parameter(2.0).variable()
    calls = []

    def declaration():
        calls.append(True)
        return parameter >= 1.0

    program, descriptors, _ = System()._compile_constraints((declaration,), {}, {id(parameter): 0})
    assert calls == [True] and descriptors[0].lower_bound == 1.0
    parameter.fixed()
    assert program.evaluate([3.0]) == (3.0,)


@pytest.mark.parametrize("invalid", [object(), True, None, lambda: 1.0, lambda context: context])
def test_malformed_constraints_fail_normalization(invalid):
    with pytest.raises(TypeError):
        System()._compile_constraints((invalid,), {}, {})


def test_generated_gap_constraints_are_collected_and_detect_reversed_fixed_layout():
    system = System(Space(d=0.2), optical_input=Ray(x=0, theta=0, n=1.0)).add(Plane(), z=0.1)
    bindings = system._collect_bindings()
    placements, _ = system._resolve_layout(system._resolve_elements(), bindings)
    constraints = system._collect_constraints(placements)
    assert len(constraints) == 2 and constraints[1] is placements[1].element.constraints[0]
    with pytest.raises(ValueError, match="unsatisfiable constant constraint"):
        system._compile_constraints(constraints, bindings, {})
    assert len(system.constraints) == 1


def test_variable_gap_constraint_remains_live_for_later_values():
    distance = Parameter(0.1).variable()
    system = System(Space(d=distance), optical_input=Ray(x=0, theta=0, n=1.0)).add(Plane(), z=0.2)
    bindings = system._collect_bindings()
    placements, _ = system._resolve_layout(system._resolve_elements(), bindings)
    program, descriptors, _ = system._compile_constraints(system._collect_constraints(placements), bindings, {id(distance): 0})
    np.testing.assert_allclose(program.evaluate([0.1]), [0.1, 0.1])
    np.testing.assert_allclose(program.evaluate([0.3]), [0.3, -0.1])
    assert [data.lower_bound for data in descriptors] == [0.0, 0.0]


def test_build_handoff_retains_order_and_does_not_mutate_declarations():
    distance = Parameter(0.1).variable()
    space, source = Space(d=distance), Ray(x=0, theta=0, n=1.0)
    graph_constraint = Constraint(distance, upper_bound=0.5, label="system limit")
    deferred = Constraint(lambda result: 1.0 if result.final.x > 0 else -1.0, label="deferred")
    system = System(space, optical_input=source).require(graph_constraint, deferred)
    before_system, before_element = system.system_constraints, space.constraints
    first, second = system.build(), system.build()
    assert [data.label for data in first._constraint_descriptors] == ["system limit", None]
    assert [data.position for data in first._constraint_descriptors] == [0, 2]
    assert first._callable_constraints[0][0] == 1 and first._callable_constraints[0][1] is deferred
    assert len(first._constraint_descriptors) == len(second._constraint_descriptors)
    assert system.system_constraints == before_system and space.constraints == before_element
    assert space.d.node is distance and distance.value == 0.1 and distance.is_variable


@pytest.mark.parametrize("owner", ["system", "element"])
def test_malformed_attached_constraint_is_rejected_during_build(owner):
    space = Space(d=0.1)
    system = System(space, optical_input=Ray(x=0, theta=0, n=1.0))
    (system if owner == "system" else space).require(object())
    with pytest.raises(TypeError, match="Constraint"):
        system.build()


def test_metadata_holds_the_parameter_and_element_objects_it_indexes_by_id():
    """Equality and hashing of the metadata must not depend on these references."""
    import gc
    import weakref

    distance = Parameter(0.1).variable()
    space = Space(d=distance)
    system = System(space, optical_input=Ray(x=0, theta=0, n=1.0))
    program, _, _, parameters, metadata, _ = compile_optics(system)
    references = weakref.ref(distance), weakref.ref(space)
    held_parameters, held_metadata = parameters, metadata
    del distance, space, system
    gc.collect()
    assert all(reference() is not None for reference in references)
    assert held_parameters[0].parameter is references[0]() and held_metadata[0].element is references[1]()
    assert hash(held_parameters[0]) == hash(ParameterData(
        held_parameters[0].parameter_id, held_parameters[0].name, held_parameters[0].value,
        held_parameters[0].lower_bound, held_parameters[0].upper_bound, held_parameters[0].parameter_index))
    assert program.parameter_indices == {held_parameters[0].parameter_id: 0}
