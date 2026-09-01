from dataclasses import dataclass
from typing import ClassVar

import pytest
import autograd.numpy as np

from autograd import grad

from diffractix.beams.base import ParaxialState
from diffractix.simulation import Simulation, SimulationResult
from diffractix.simulation.simulation import SimulationStep


class DummyResult:
    pass


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
    parameter_info=(),
    location_map=None,
    requirements=(),
):
    if source is None:
        source = DummyState()

    if location_map is None:
        location_map = {}

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
        requirements=requirements,
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
    parameter_info = (object(),)
    location_map = {1: ((0, 1),)}
    requirements = (object(),)

    simulation = Simulation(
        source=source,
        graph=graph,
        steps=[step],
        parameter_info=parameter_info,
        location_map=location_map,
        requirements=requirements,
    )

    assert simulation.source is source
    assert simulation.graph is graph
    assert simulation.steps == (step,)
    assert simulation.parameter_info == parameter_info
    assert simulation.location_map is location_map
    assert simulation.requirements == requirements


def test_simulation_converts_sequences_to_tuples():
    simulation = Simulation(
        source=DummyState(),
        graph=DummyGraph(
            initial_values=np.array([]),
            evaluator=lambda theta: np.array([]),
        ),
        steps=[],
        parameter_info=[],
        location_map={},
        requirements=[],
    )

    assert simulation.steps == ()
    assert simulation.parameter_info == ()
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
        parameter_info=(),
        location_map={},
    )

    assert simulation.initial_values is initial_values


def test_simulation_rejects_invalid_result_type():
    @dataclass(frozen=True)
    class InvalidState(ParaxialState):
        result_type: ClassVar = None

        def propagate(self, A, B, C, D, n):
            return self

    graph = DummyGraph(
        initial_values=np.array([]),
        evaluator=lambda theta: np.array([]),
    )

    with pytest.raises(TypeError, match="result_type must be a type"):
        Simulation(
            source=InvalidState(),
            graph=graph,
            steps=(),
            parameter_info=(),
            location_map={},
        )


# -------
# RESULTS
# -------

def test_run_returns_simulation_result():
    simulation = create_simulation(
        values=(),
        steps=(),
    )

    result = simulation.run()

    assert isinstance(result, SimulationResult)


def test_run_returns_source_specific_result_type():
    simulation = create_simulation(
        values=(),
        steps=(),
    )

    result = simulation.run()

    assert isinstance(result, DummyResult)


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
        location: (0, 0),
    }

    simulation = create_simulation(
        values=(),
        steps=(),
        location_map=location_map,
    )

    result = simulation.run()

    assert result.at(location) is result.initial


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
        parameter_info=(),
        location_map={},
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
        parameter_info=(),
        location_map={},
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
        parameter_info=(),
        location_map={},
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
        parameter_info=(),
        location_map={},
    )

    def objective(value):
        result = simulation.run(np.array([value]))
        return result.states[-1].value

    derivative = grad(objective)(2.0)

    assert derivative == pytest.approx(2.0)