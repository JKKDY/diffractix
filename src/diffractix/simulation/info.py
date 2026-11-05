from __future__ import annotations

from dataclasses import dataclass, replace

from diffractix.input.paraxial_input import ParaxialState


@dataclass(frozen=True)
class SimulationStep:
    """Numerical lookup information for one propagation step."""

    matrix_indices: tuple[tuple[int, int], tuple[int, int]]
    length_index: int
    refractive_index_index: int


@dataclass(frozen=True)
class SimulationInput:
    """Construct a numerical input state from compiled graph outputs."""

    initial_state: ParaxialState
    field_indices: tuple[tuple[str, int], ...]

    def construct(self, values):
        updates = {
            name: values[index]
            for name, index in self.field_indices
        }
        return replace(self.initial_state, **updates)
