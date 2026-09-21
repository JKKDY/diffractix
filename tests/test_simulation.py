from dataclasses import dataclass
from typing import ClassVar

import pytest
import autograd.numpy as np

from autograd import grad

from diffractix.beams import GaussianBeam, ParaxialRay, RayBundle
from diffractix.beams.base import ParaxialState
from diffractix.simulation import Simulation, SimulationResult
from diffractix.simulation.simulation import SimulationStep
from diffractix.system.info import ElementInfo


class DummyResult:
    pass


def element_info_for_steps(steps):
    return tuple(
        ElementInfo(
            type_name="DummyElement",
            label=None,
            path=None,
            parameter_names=(),
            parameter_indices=(),
        )
        for _ in steps
    )


def element_info_for_states(states):
    return element_info_for_steps(states[1:])


def create_result(**kwargs):
    kwargs.setdefault(
        "element_info",
        element_info_for_states(kwargs["states"]),
    )
    return SimulationResult(**kwargs)


@dataclass(frozen=True)
class DummyState(ParaxialState):
    result_type: ClassVar[type] = DummyResult

    value: float = 0.0
    calls: tuple = ()

    def propagate(self, A, B, C, D, n):
        return DummyState(
            value=self.value + A + 2 * B + 3 * C + 4 * D + 5 * n,
            calls=self.calls + ((A, B, C, D, n),),
        )


@dataclass
class DummyGraph:
    initial_values: np.ndarray
    evaluator: callable

    def evaluate(self, theta):
        return self.evaluator(theta)


def create_simulation(
    values,
    steps,
    *,
    initial_values=(),
    source=None,
    parameter_info=None,
    location_map=None,
    requirements=(),
    element_info=None,
):
    if source is None:
        source = DummyState()

    if location_map is None:
        location_map = {}

    if parameter_info is None:
        parameter_info = {}

    if element_info is None:
        element_info = element_info_for_steps(steps)

    graph = DummyGraph(
        initial_values=np.array(initial_values),
        evaluator=lambda theta: np.array(values),
    )

    return Simulation(
        source=source,
        graph=graph,
        steps=steps,
        parameter_info=parameter_info,
        location_map=location_map,
        simulation_context={},
        requirements=requirements,
        parameter_graph = None,
        element_info=element_info,
    )


# --------------
# INITIALIZATION
# --------------

def test_simulation_stores_compiled_data():
    source = DummyState()
    graph = DummyGraph(
        initial_values=np.array([1.0]),
        evaluator=lambda theta: np.array([]),
    )
    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )
    parameter_info = {1: object()}
    location_map = {1: ((0, 1),)}
    requirements = (object(),)
    element_info = ElementInfo(
        type_name="Space",
        label="Drift",
        path=None,
        parameter_names=("d",),
        parameter_indices=(0,),
    )

    simulation = Simulation(
        source=source,
        graph=graph,
        steps=[step],
        parameter_info=parameter_info,
        location_map=location_map,
        simulation_context={},
        requirements=requirements,
        parameter_graph=graph,
        element_info=[element_info],
    )

    assert simulation.source is source
    assert simulation.graph is graph
    assert simulation.steps == (step,)
    assert simulation.parameter_info == parameter_info
    assert simulation.location_map is location_map
    assert simulation.requirements == requirements
    assert simulation.parameter_graph is graph
    assert simulation.element_info == (element_info,)


def test_simulation_converts_sequence_fields_to_tuples():
    simulation = Simulation(
        source=DummyState(),
        graph=DummyGraph(
            initial_values=np.array([]),
            evaluator=lambda theta: np.array([]),
        ),
        steps=[],
        parameter_info={},
        location_map={},
        simulation_context={},
        requirements=[],
        parameter_graph=None,
        element_info=(),
    )

    assert simulation.steps == ()
    assert simulation.parameter_info == {}
    assert simulation.requirements == ()


def test_simulation_initial_values_are_graph_initial_values():
    initial_values = np.array([1.0, 2.0])
    graph = DummyGraph(
        initial_values=initial_values,
        evaluator=lambda theta: np.array([]),
    )

    simulation = Simulation(
        source=DummyState(),
        graph=graph,
        steps=(),
        parameter_info={},
        location_map={},
        simulation_context={},
        requirements=(),
        parameter_graph=None,
        element_info=(),
    )

    assert simulation.initial_values is initial_values


def test_simulation_rejects_non_dataclass_state():
    class InvalidState(ParaxialState):
        def propagate(self, A, B, C, D, n):
            return self
    graph = DummyGraph(
        initial_values=np.array([]),
        evaluator=lambda theta: np.array([]),
    )
    with pytest.raises(TypeError, match="must be a dataclass"):
        Simulation(
            source=InvalidState(),
            graph=graph,
            steps=(),
            parameter_info={},
            location_map={},
            simulation_context = {},
            requirements=(),
            parameter_graph = None,
            element_info = ()
        )


# -------
# RESULTS
# -------

def test_paraxial_state_has_no_default_result_columns():
    assert ParaxialState.result_columns == ()


def test_concrete_state_result_columns_reference_existing_properties():
    states = (
        GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6),
        ParaxialRay(x=1.0, theta=0.1),
        RayBundle(x=np.array([0.0, 1.0]), theta=np.array([0.0, 0.1])),
    )

    assert GaussianBeam.result_columns == ("w", "R", "gouy_phase")
    assert ParaxialRay.result_columns == ("x", "theta")
    assert RayBundle.result_columns == ("x", "theta")

    for state in states:
        assert state.result_columns
        assert all(hasattr(state, name) for name in state.result_columns)

def test_run_returns_simulation_result():
    simulation = create_simulation(
        values=(),
        steps=(),
    )

    result = simulation.run()

    assert isinstance(result, SimulationResult)


def test_run_returns_generated_source_specific_result_type():
    simulation = create_simulation(
        values=(),
        steps=(),
    )
    result = simulation.run()
    assert isinstance(result, SimulationResult)
    assert hasattr(type(result), "value")


def test_run_records_source_as_initial_state():
    source = DummyState(value=3.0)

    simulation = create_simulation(
        values=(),
        steps=(),
        source=source,
    )

    result = simulation.run()

    assert result.source is source
    assert result.states == (source,)
    assert result.states[0] is source


def test_run_starts_at_zero_position():
    simulation = create_simulation(
        values=(),
        steps=(),
    )

    result = simulation.run()

    assert len(result.z) == 1
    assert result.z[0] == pytest.approx(0.0)


def test_run_forwards_location_map_to_result():
    location = object()
    location_map = {
        id(location): ((0, 0),),
    }

    simulation = create_simulation(
        values=(),
        steps=(),
        location_map=location_map,
    )

    result = simulation.run()

    assert result.at(location) is result.initial


def test_result_rejects_inconsistent_element_info_length():
    states = (DummyState(), DummyState())

    with pytest.raises(ValueError, match="one input state plus one state"):
        SimulationResult(
            source=states[0],
            z=np.array([0.0, 1.0]),
            states=states,
            location_map={},
            element_info=(),
        )

def test_result_at_unique_element_does_not_require_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0]),
        states=states,
        location_map={
            id(element): ((0, 1),),
        },
    )

    assert result.at(element) is states[0]


def test_result_after_unique_element_does_not_require_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0]),
        states=states,
        location_map={
            id(element): ((0, 1),),
        },
    )

    assert result.after(element) is states[1]


def test_result_z_accessors_return_unique_element_positions():
    element = object()
    states = (DummyState(), DummyState())
    result = create_result(
        source=states[0],
        z=np.array([1.25, 2.5]),
        states=states,
        location_map={id(element): ((0, 1),)},
    )

    assert result.z_at(element) == pytest.approx(1.25)
    assert result.z_after(element) == pytest.approx(2.5)


def test_result_at_selects_repeated_element_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
        DummyState(value=2.0),
        DummyState(value=3.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0, 0.0, 0.0]),
        states=states,
        location_map={
            id(element): (
                (0, 1),
                (2, 3),
            ),
        },
    )

    assert result.at(element, occurrence=0) is states[0]
    assert result.at(element, occurrence=1) is states[2]


def test_result_after_selects_repeated_element_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
        DummyState(value=2.0),
        DummyState(value=3.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0, 0.0, 0.0]),
        states=states,
        location_map={
            id(element): (
                (0, 1),
                (2, 3),
            ),
        },
    )

    assert result.after(element, occurrence=0) is states[1]
    assert result.after(element, occurrence=1) is states[3]


def test_result_z_accessors_select_repeated_element_occurrence():
    element = object()
    states = tuple(DummyState() for _ in range(4))
    result = create_result(
        source=states[0],
        z=np.array([0.0, 1.0, 3.0, 6.0]),
        states=states,
        location_map={id(element): ((0, 1), (2, 3))},
    )

    assert result.z_at(element, occurrence=0) == pytest.approx(0.0)
    assert result.z_after(element, occurrence=0) == pytest.approx(1.0)
    assert result.z_at(element, occurrence=1) == pytest.approx(3.0)
    assert result.z_after(element, occurrence=1) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="occurs 2 times"):
        result.z_at(element)


def test_result_at_repeated_element_requires_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
        DummyState(value=2.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0, 0.0]),
        states=states,
        location_map={
            id(element): (
                (0, 1),
                (1, 2),
            ),
        },
    )

    with pytest.raises(ValueError, match="occurs 2 times"):
        result.at(element)


def test_result_after_repeated_element_requires_occurrence():
    element = object()
    states = (
        DummyState(value=0.0),
        DummyState(value=1.0),
        DummyState(value=2.0),
    )

    result = create_result(
        source=states[0],
        z=np.array([0.0, 0.0, 0.0]),
        states=states,
        location_map={
            id(element): (
                (0, 1),
                (1, 2),
            ),
        },
    )

    with pytest.raises(ValueError, match="occurs 2 times"):
        result.after(element)


def test_result_rejects_out_of_range_occurrence():
    element = object()

    result = create_result(
        source=DummyState(),
        z=np.array([0.0]),
        states=(DummyState(),),
        location_map={
            id(element): ((0, 0),),
        },
    )

    with pytest.raises(IndexError, match="out of range"):
        result.at(element, occurrence=1)


def test_result_rejects_negative_occurrence():
    element = object()

    result = create_result(
        source=DummyState(),
        z=np.array([0.0]),
        states=(DummyState(),),
        location_map={
            id(element): ((0, 0),),
        },
    )

    with pytest.raises(IndexError, match="out of range"):
        result.at(element, occurrence=-1)


def test_result_rejects_non_integer_occurrence():
    element = object()

    result = create_result(
        source=DummyState(),
        z=np.array([0.0]),
        states=(DummyState(),),
        location_map={
            id(element): ((0, 0),),
        },
    )

    with pytest.raises(TypeError, match="occurrence must be an integer"):
        result.at(element, occurrence=1.5)


def test_result_rejects_occurrence_for_numeric_position():
    result = create_result(
        source=DummyState(),
        z=np.array([0.0]),
        states=(DummyState(),),
        location_map={},
        probe=lambda z: DummyState(value=z),
    )

    with pytest.raises(TypeError, match="occurrence may only be specified"):
        result.at(0.5, occurrence=0)


@pytest.mark.parametrize("accessor", ["z_at", "z_after"])
def test_result_z_accessors_match_location_lookup_errors(accessor):
    element = object()
    missing = object()
    result = create_result(
        source=DummyState(),
        z=np.array([0.0]),
        states=(DummyState(),),
        location_map={id(element): ((0, 0),)},
    )
    resolve = getattr(result, accessor)

    with pytest.raises(KeyError, match="not part of this simulation"):
        resolve(missing)
    with pytest.raises(IndexError, match="out of range"):
        resolve(element, occurrence=1)


# -----------
# PROPAGATION
# -----------

def test_run_propagates_single_step():
    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = create_simulation(
        values=(
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
            1.5,
        ),
        steps=(step,),
    )

    result = simulation.run()

    assert len(result.states) == 2
    assert result.states[1].calls == (
        (1.0, 2.0, 3.0, 4.0, 1.5),
    )


def test_run_reads_values_using_step_indices():
    step = SimulationStep(
        matrix_indices=((4, 1), (5, 0)),
        length_index=3,
        refractive_index_index=2,
    )

    simulation = create_simulation(
        values=(
            10.0,
            20.0,
            1.5,
            0.25,
            30.0,
            40.0,
        ),
        steps=(step,),
    )

    result = simulation.run()

    assert result.states[1].calls == (
        (30.0, 20.0, 40.0, 10.0, 1.5),
    )


def test_run_propagates_steps_sequentially():
    first = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )
    second = SimulationStep(
        matrix_indices=((6, 7), (8, 9)),
        length_index=10,
        refractive_index_index=11,
    )

    simulation = create_simulation(
        values=(
            1.0, 2.0, 3.0, 4.0, 0.1, 1.0,
            5.0, 6.0, 7.0, 8.0, 0.2, 1.5,
        ),
        steps=(first, second),
    )

    result = simulation.run()

    assert len(result.states) == 3
    assert result.states[1].calls == (
        (1.0, 2.0, 3.0, 4.0, 1.0),
    )
    assert result.states[2].calls == (
        (1.0, 2.0, 3.0, 4.0, 1.0),
        (5.0, 6.0, 7.0, 8.0, 1.5),
    )


def test_run_records_state_after_every_step():
    first = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )
    second = SimulationStep(
        matrix_indices=((6, 7), (8, 9)),
        length_index=10,
        refractive_index_index=11,
    )

    simulation = create_simulation(
        values=(
            1.0, 0.0, 0.0, 1.0, 0.1, 1.0,
            1.0, 0.0, 0.0, 1.0, 0.2, 1.0,
        ),
        steps=(first, second),
    )

    result = simulation.run()

    assert len(result.states) == len(simulation.steps) + 1


# ---------
# POSITION
# ---------

def test_run_accumulates_element_lengths():
    first = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )
    second = SimulationStep(
        matrix_indices=((6, 7), (8, 9)),
        length_index=10,
        refractive_index_index=11,
    )

    simulation = create_simulation(
        values=(
            1.0, 0.0, 0.0, 1.0, 0.25, 1.0,
            1.0, 0.0, 0.0, 1.0, 0.40, 1.0,
        ),
        steps=(first, second),
    )

    result = simulation.run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.25, 0.65]),
    )


def test_run_preserves_duplicate_positions_for_zero_length_step():
    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = create_simulation(
        values=(
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ),
        steps=(step,),
    )

    result = simulation.run()

    assert np.allclose(
        result.z,
        np.array([0.0, 0.0]),
    )


def test_run_positions_align_with_states():
    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = create_simulation(
        values=(
            1.0,
            0.0,
            0.0,
            1.0,
            0.1,
            1.0,
        ),
        steps=(step,),
    )

    result = simulation.run()

    assert len(result.z) == len(result.states)


# -----
# THETA
# -----

def test_run_uses_initial_values_when_theta_is_none():
    graph = DummyGraph(
        initial_values=np.array([2.0]),
        evaluator=lambda theta: np.array([
            1.0,
            theta[0],
            0.0,
            1.0,
            0.0,
            1.0,
        ]),
    )

    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = Simulation(
        source=DummyState(),
        graph=graph,
        steps=(step,),
        parameter_info={},
        location_map={},
        simulation_context={},
        requirements=(),
        parameter_graph=None,
        element_info=element_info_for_steps((step,)),
    )

    result = simulation.run()

    assert result.states[-1].calls[0][1] == pytest.approx(2.0)


def test_run_uses_supplied_theta():
    graph = DummyGraph(
        initial_values=np.array([2.0]),
        evaluator=lambda theta: np.array([
            1.0,
            theta[0],
            0.0,
            1.0,
            0.0,
            1.0,
        ]),
    )

    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = Simulation(
        source=DummyState(),
        graph=graph,
        steps=(step,),
        parameter_info={},
        location_map={},
        simulation_context={},
        requirements=(),
        parameter_graph=None,
        element_info=element_info_for_steps((step,)),
    )

    result = simulation.run(np.array([7.0]))

    assert result.states[-1].calls[0][1] == pytest.approx(7.0)


def test_run_does_not_modify_initial_values():
    graph = DummyGraph(
        initial_values=np.array([2.0]),
        evaluator=lambda theta: np.array([
            1.0,
            theta[0],
            0.0,
            1.0,
            0.0,
            1.0,
        ]),
    )

    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = Simulation(
        source=DummyState(),
        graph=graph,
        steps=(step,),
        parameter_info={},
        location_map={},
        simulation_context={},
        requirements=(),
        parameter_graph=None,
        element_info=element_info_for_steps((step,)),
    )

    simulation.run(np.array([7.0]))

    assert np.allclose(
        simulation.initial_values,
        np.array([2.0]),
    )


# ------
# PURITY
# ------

def test_run_does_not_mutate_source_state():
    source = DummyState(value=3.0)

    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = create_simulation(
        values=(
            1.0,
            2.0,
            3.0,
            4.0,
            0.1,
            1.0,
        ),
        steps=(step,),
        source=source,
    )

    simulation.run()

    assert source.value == pytest.approx(3.0)
    assert source.calls == ()


def test_repeated_runs_are_independent():
    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = create_simulation(
        values=(
            1.0,
            2.0,
            3.0,
            4.0,
            0.1,
            1.0,
        ),
        steps=(step,),
    )

    first = simulation.run()
    second = simulation.run()

    assert first.states[0] is simulation.source
    assert second.states[0] is simulation.source
    assert first.states[1] is not second.states[1]
    assert first.states[1] == second.states[1]


# -----------------
# DIFFERENTIABILITY
# -----------------

def test_run_is_differentiable_with_respect_to_theta():
    graph = DummyGraph(
        initial_values=np.array([2.0]),
        evaluator=lambda theta: np.array([
            1.0,
            theta[0],
            0.0,
            1.0,
            0.0,
            1.0,
        ]),
    )

    step = SimulationStep(
        matrix_indices=((0, 1), (2, 3)),
        length_index=4,
        refractive_index_index=5,
    )

    simulation = Simulation(
        source=DummyState(),
        graph=graph,
        steps=(step,),
        parameter_info={},
        location_map={},
        simulation_context = {},
        requirements=(),
        parameter_graph=None,
        element_info=element_info_for_steps((step,)),
    )

    def objective(value):
        result = simulation.run(np.array([value]))
        return result.states[-1].value

    derivative = grad(objective)(2.0)

    assert derivative == pytest.approx(2.0)
