from __future__ import annotations

import autograd.numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .beams import ParaxialState
from .graph import CompiledAST
from .simulation_result import SimulationResult
from .simulation_metadata import ParameterInfo


@dataclass(frozen=True)
class SimulationStep:
    """Numerical lookup information for one propagation step."""

    matrix_indices: tuple[tuple[int, int], tuple[int, int]]
    length_index: int
    refractive_index_index: int


class Simulation:
    """
    Compiled differentiable optical simulation.

    A Simulation contains the numerical program and metadata required to
    evaluate an optical system. Running a simulation does not mutate the source
    System, its elements, or parameter graph.
    """

    def __init__(
        self,
        source: ParaxialState,
        graph: CompiledAST,
        steps: Sequence[SimulationStep],
        parameter_info: Sequence[ParameterInfo],
        location_map: Mapping,
        requirements=(),
    ):
        self.source = source
        self.graph = graph
        self.steps = tuple(steps)
        self.parameter_info = tuple(parameter_info)
        self.location_map = location_map
        self.requirements = tuple(requirements)

        result_type = self.source.result_type

        if not isinstance(result_type, type):
            raise TypeError(
                f"{type(self.source).__name__}.result_type must be a type, "
                f"got {type(result_type).__name__}."
            )

        self._result_type = type(
            f"{type(self.source).__name__}SimulationResult",
            (result_type, SimulationResult),
            {},
        )

    @property
    def initial_values(self):
        return self.graph.initial_values

    def run(self, theta=None) -> SimulationResult:
        """
        Run the optical simulation.

        Parameters
        ----------
        theta:
            Optional independent parameter vector. If None, the compiled initial
            values are used.

        Returns
        -------
        SimulationResult
            Rich numerical trace of the propagated optical state.
        """
        if theta is None:
            theta = self.initial_values

        values = self.graph.evaluate(theta)

        state = self.source
        z = 0.0

        states = [state]
        positions = [z]

        for step in self.steps:
            A = values[step.matrix_indices[0][0]]
            B = values[step.matrix_indices[0][1]]
            C = values[step.matrix_indices[1][0]]
            D = values[step.matrix_indices[1][1]]
            length = values[step.length_index]
            n = values[step.refractive_index_index]

            state = state.propagate(A, B, C, D, n)
            z = z + length

            states.append(state)
            positions.append(z)

        return self._result_type(
            source=self.source,
            z=np.stack(positions),
            states=tuple(states),
            location_map=self.location_map,
        )