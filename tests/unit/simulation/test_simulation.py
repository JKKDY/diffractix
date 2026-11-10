"""Compiled optical execution, numerical source reconstruction and metadata."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.graph import Literal, Node, Parameter, Program, compile_graph
from diffractix.input import GaussianBeam, ParaxialState, Ray, RayBundle
from diffractix.simulation.info import SimulationInput, SimulationStep
from diffractix.simulation.result import SimulationResult
from diffractix.simulation.simulation import Simulation
from diffractix.solver.constraint import Constraint
from diffractix.system.info import ElementData, ParameterData
from diffractix.solver.constraint import ConstraintDescriptor


@dataclass(eq=False)
class RecordingState(ParaxialState):
    value: float = 0.0
    calls: tuple = ()

    def propagate(self, A, B, C, D, n):
        return type(self)(
            value=self.value + A + 2*B + 3*C + 4*D + 5*n,
            calls=(*self.calls, (A, B, C, D, n)),
        )


def make_simulation(roots=(), steps=(), *, source=None, field_indices=(), parameters=()):
    roots = tuple(value if isinstance(value, Node) else Literal(value) for value in roots)
    program = compile_graph(roots)
    parameters_by_id = {id(parameter): parameter for parameter in parameters}
    parameter_data = tuple(
        ParameterData(pid, parameters_by_id[pid].name, parameters_by_id[pid].value,
                      parameters_by_id[pid].lower_bound, parameters_by_id[pid].upper_bound, index)
        for pid, index in sorted(program.parameter_indices.items(), key=lambda item: item[1])
    )
    elements = tuple(
        ElementData(index, index, "CustomElement", f"Stage {index}", None, frozenset())
        for index in range(len(steps))
    )
    source = RecordingState.numerical_type()() if source is None else source
    return Simulation(
        element_data=elements, parameter_data=parameter_data,
        callable_constraints=(), constraint_descriptors=(),
        constraint_program=compile_graph(()), optics_program=program,
        optical_stages=steps,
        optical_inputs=(SimulationInput(source, tuple(field_indices)),),
    )


def test_constructor_owns_sequence_containers_and_retains_programs():
    simulation = make_simulation()
    elements, parameters, sources, stages = [], [], list(simulation.optical_inputs), []
    descriptors = [ConstraintDescriptor(0, 0, 1, "graph", "bounded graph")]
    deferred = [(1, Constraint(lambda result: result.final.value, label="deferred"))]
    copied = Simulation(
        element_data=elements, parameter_data=parameters,
        callable_constraints=deferred, constraint_descriptors=descriptors,
        constraint_program=simulation.constraint_program, optics_program=simulation.optics_program,
        optical_stages=stages, optical_inputs=sources,
    )
    sources.clear()
    deferred.clear()
    descriptors.clear()
    elements.append(object())
    parameters.append(object())
    stages.append(object())

    assert copied.optics_program is simulation.optics_program
    assert copied.constraint_program is simulation.constraint_program
    assert copied.element_data == copied.parameter_data == copied.optical_stages == ()
    assert len(copied.optical_inputs) == len(copied.callable_constraints) == len(copied.constraint_descriptors) == 1
    assert all(isinstance(value, tuple) for value in (
        copied.element_data, copied.parameter_data, copied.optical_inputs,
        copied.optical_stages, copied.callable_constraints, copied.constraint_descriptors,
    ))


def test_zero_stage_run_returns_an_independent_numerical_source():
    source = RecordingState.numerical_type()(value=3.0)
    simulation = make_simulation(source=source)

    result = simulation.run()

    assert isinstance(result, SimulationResult)
    assert result.source is result.initial is result.final
    assert result.source is not source
    assert result.source.value == source.value == 3.0
    assert result.states == (result.source,)
    np.testing.assert_array_equal(result.z, [0.0])
    assert result.optical_values == ()


@pytest.mark.parametrize("count", [0, 2])
def test_run_requires_exactly_one_optical_input(count):
    simulation = make_simulation()
    simulation.optical_inputs = simulation.optical_inputs * count

    with pytest.raises(ValueError, match="exactly one optical input"):
        simulation.run()


def test_run_evaluates_the_program_once_and_obeys_noncontiguous_step_indices(monkeypatch):
    step = SimulationStep(((4, 1), (5, 0)), length_index=3, refractive_index_index=2)
    simulation = make_simulation((10., 20., 1.5, .25, 30., 40.), (step,))
    calls = []
    evaluate = Program.evaluate

    def counted(program, theta, bindings={}):
        calls.append((program, theta))
        return evaluate(program, theta, bindings)

    monkeypatch.setattr(Program, "evaluate", counted)
    result = simulation.run()

    assert len(calls) == 1 and calls[0][0] is simulation.optics_program
    assert result.final.calls == ((30., 20., 40., 10., 1.5),)
    assert result.final.value == 237.5
    np.testing.assert_allclose(result.z, [0, .25])
    assert result.optical_values == (10., 20., 1.5, .25, 30., 40.)
    assert result.element_data is simulation.element_data
    assert result.optical_stages is simulation.optical_stages


def test_steps_propagate_sequentially_and_keep_zero_length_boundary_samples():
    steps = (SimulationStep(((0, 1), (2, 3)), 4, 5),
             SimulationStep(((6, 7), (8, 9)), 10, 11))
    source = RecordingState.numerical_type()(value=3.)
    simulation = make_simulation((1., 2., 3., 4., .1, 1., 5., 6., 7., 8., 0., 1.5), steps, source=source)

    result = simulation.run()

    assert result.states[1].calls == ((1., 2., 3., 4., 1.),)
    assert result.final.calls == ((1., 2., 3., 4., 1.), (5., 6., 7., 8., 1.5))
    assert result.states[1].value == 38.
    assert result.final.value == 115.5
    assert source.value == 3. and source.calls == ()
    np.testing.assert_allclose(result.z, [0., .1, .1])


def test_initial_guesses_are_independent_of_defaults_and_live_parameters():
    distance = Parameter(.1, name="distance").variable(bounds=(0., 1.))
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation((1., distance, 0., 1., distance, 1.), (step,), parameters=(distance,))
    original = simulation.initial_values
    changed = simulation.initial_values
    changed[distance] = .4
    distance.value = .9
    distance.fixed()

    assert changed[distance] == .4
    assert original[distance] == .1
    np.testing.assert_allclose(simulation.initial_values.values, [.1])
    assert not simulation._initial_values.flags.writeable
    with pytest.raises(ValueError):
        simulation._initial_values[0] = .3
    assert simulation.run().final.calls[0][1] == .1
    assert simulation.run(changed.values).final.calls[0][1] == .4
    assert distance.value == .9 and not distance.is_variable


def test_repeated_runs_do_not_reuse_mutable_state_objects_or_change_theta():
    parameter = Parameter(2.).variable()
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation((1., parameter, 0., 1., .2, 1.), (step,), parameters=(parameter,))
    theta = np.array([7.])

    first, second = simulation.run(theta), simulation.run()

    assert first.source is not second.source
    assert first.final is not second.final
    assert first.final.calls[0][1] == 7.
    assert second.final.calls[0][1] == 2.
    np.testing.assert_array_equal(theta, [7.])
    np.testing.assert_array_equal(simulation.initial_values.values, [2.])
    assert parameter.value == 2.


def test_source_reconstruction_uses_outputs_and_recomputes_derived_fields():
    @dataclass(eq=False)
    class DerivedState(ParaxialState):
        value: float = 2.
        doubled: float = field(init=False)

        def __post_init__(self):
            self.doubled = 2*self.value

        def propagate(self, A, B, C, D, n):
            return type(self)(value=A*self.value + B)

    source = DerivedState.numerical_type()(value=2.)
    source.doubled = -99.
    specification = SimulationInput(source, (("value", 1),))

    reconstructed = specification.construct((100., 5.))

    assert type(reconstructed) is type(source)
    assert reconstructed.value == 5. and reconstructed.doubled == 10.
    assert reconstructed.parameter_names == ()
    assert source.value == 2. and source.doubled == -99.


def test_native_roots_keep_wavelength_real_beside_complex_propagation_coefficients():
    wavelength = Parameter(1e-6, name="wavelength").variable()
    aperture = -1j*wavelength / (np.pi * 1e-6)
    source = GaussianBeam.numerical_type()(q=.02j, wavelength=1e-6, n=1.)
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation(
        (1., 0., aperture, 1., 0., 1., wavelength), (step,), source=source,
        field_indices=(("wavelength", 6),), parameters=(wavelength,),
    )

    result = simulation.run(xp.array([.8e-6]))

    assert not np.iscomplexobj(result.source.wavelength)
    assert np.iscomplexobj(result.optical_values[2])
    expected_q = .02j / (1 + (-1j*.8e-6/(np.pi*1e-6))*.02j)
    assert result.final.q == pytest.approx(expected_q)
    assert result.source.wavelength == result.final.wavelength == .8e-6

    width_squared = lambda theta: simulation.run(theta).final.w**2
    coefficient = .02/(np.pi*1e-6)
    denominator = 1 + coefficient*.8e-6
    for function, expected in (
        (grad(width_squared), [.02/(np.pi*denominator**2)]),
        (hessian(width_squared), [[-2*.02*coefficient/(np.pi*denominator**3)]]),
    ):
        np.testing.assert_allclose(function(xp.array([.8e-6])), expected, rtol=1e-11)
        np.testing.assert_allclose(jit(function)(xp.array([.8e-6])), expected, rtol=1e-11)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_runtime_source_array_dtype_survives_scalar_roots(dtype):
    source = RayBundle.numerical_type()(
        x=xp.array([1e-3, -2e-3], dtype=dtype), theta=xp.array([.01, .02], dtype=dtype), n=1.,
    )
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation((1., .1, 0., 1., .1, 1.), (step,), source=source)
    result = simulation.run()
    np.testing.assert_allclose(result.final.x, [2e-3, 0.], atol=3e-10)
    assert result.final.x.dtype == np.dtype(dtype)


def test_source_and_optics_share_theta_with_analytical_derivatives_and_jit():
    distance, initial_x = Parameter(.2).variable(), Parameter(.001).variable()
    source = Ray.numerical_type()(x=.001, theta=.01, n=1.)
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation(
        (1., distance, 0., 1., distance, 1., initial_x), (step,), source=source,
        field_indices=(("x", 6),), parameters=(distance, initial_x),
    )

    def outputs(theta):
        result = simulation.run(theta)
        return xp.stack((result.source.x, result.final.x, result.z[-1]))

    def cost(theta):
        return simulation.run(theta).final.x**2

    point = xp.array([.3, .002])
    direction = np.array([.01, 1.])
    for function, expected in (
        (outputs, [.002, .005, .3]),
        (jacobian(outputs), [[0., 1.], [.01, 1.], [1., 0.]]),
        (grad(cost), .01*direction),
        (hessian(cost), 2*np.outer(direction, direction)),
        (grad(lambda t: xp.sum(grad(cost)(t)**2)), 8*.005*np.dot(direction, direction)*direction),
    ):
        np.testing.assert_allclose(function(point), expected, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(jit(function)(point), expected, rtol=1e-11, atol=1e-13)


def test_string_summary_reads_frozen_values_without_mutating_them():
    step = SimulationStep(((0, 1), (2, 3)), 4, 5)
    simulation = make_simulation((1., .1, 0., 1., .1, 1.), (step,))
    before = simulation.optics_program.evaluate([])

    text = str(simulation)

    assert "Compiled Simulation" in text
    assert "Stage 0" in text and "CustomElement" in text
    assert "Total length: 0.1" in text and "Variables: 0" in text
    assert simulation.optics_program.evaluate([]) == before


def test_summary_reconstructs_interleaved_constraint_declaration_order():
    simulation = make_simulation()
    simulation.constraint_descriptors = (
        ConstraintDescriptor(0, 0., 1., "first graph", "first"),
        ConstraintDescriptor(2, 0., 1., "last graph", "last"),
    )
    simulation.callable_constraints = ((1, Constraint(lambda result: result.final.value, label="middle callable")),)

    text = str(simulation)

    assert text.index("first graph") < text.index("middle callable") < text.index("last graph")
    assert "Constraints: 3" in text


def test_result_columns_are_existing_quantities_on_each_numerical_state():
    assert ParaxialState.result_columns == ()
    states = (
        Ray.numerical_type()(x=.001, theta=.01, n=1.),
        RayBundle.numerical_type()(x=xp.array([.001]), theta=xp.array([.01]), n=1.),
        GaussianBeam.numerical_type()(q=.02j, wavelength=1e-6, n=1.),
    )
    for state in states:
        assert state.result_columns
        for name in state.result_columns:
            value = getattr(state, name)
            assert np.shape(value) in ((), (1,))


def test_symbolic_result_requires_exactly_one_optical_input():
    simulation = make_simulation()
    simulation.optical_inputs = ()
    with pytest.raises(ValueError, match="exactly one optical input"):
        simulation.symbolic
