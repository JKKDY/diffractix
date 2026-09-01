from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from numbers import Integral, Real

from ..elements import OpticalElement
from ..graph import Parameter
from ..beams.base import ParaxialState
import autograd.numpy as np

@dataclass(frozen=True)
class ParameterInfo:
    """Metadata describing a simulation parameter."""

    index: int
    parameter: Parameter
    name: str
    label: str
    owner: object | None
    initial_value: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class SimulationStep:
    """Numerical lookup information for one propagation step."""

    matrix_indices: tuple[tuple[int, int], tuple[int, int]]
    length_index: int
    refractive_index_index: int



class SimulationResult:
    """Numerical result produced by a Simulation."""

    def __init__(
        self,
        source: ParaxialState,
        z :  np.ndarray,
        states: Sequence[ParaxialState],
        location_map: Mapping[int, tuple[tuple[int, int], ...]],
        probe: Callable[[float], ParaxialState] | None = None,
    ):
        if len(z) != len(states):
            raise ValueError(
                f"z and states must have the same length, got {len(z)} and {len(states)}."
            )

        if not states:
            raise ValueError("SimulationResult requires at least one state.")

        self._source = source
        self.z = z
        self.states = tuple(states)
        self._location_map = dict(location_map)
        self._probe = probe

    @property
    def source(self) -> ParaxialState:
        return self._source

    @property
    def initial(self) -> ParaxialState:
        return self.states[0]

    @property
    def final(self) -> ParaxialState:
        return self.states[-1]

    def _resolve_location(self, element, occurrence=None):
        try:
            locations = self._location_map[id(element)]
        except KeyError:
            raise KeyError(f"Element {element!r} is not part of this simulation.") from None

        if occurrence is None:
            if len(locations) > 1:
                raise ValueError(
                    f"Element {element!r} occurs {len(locations)} times in this simulation. "
                    "Specify occurrence= to select one."
                )

            return locations[0]

        if isinstance(occurrence, bool) or not isinstance(occurrence, Integral):
            raise TypeError(f"occurrence must be an integer, got {type(occurrence).__name__}.")

        if occurrence < 0 or occurrence >= len(locations):
            raise IndexError(
                f"Element {element!r} has {len(locations)} occurrence(s); "
                f"occurrence {occurrence} is out of range."
            )

        return locations[occurrence]

    def at(self, location, occurrence=None) -> ParaxialState:
        """
        Return the optical state at a location.

        Parameters
        ----------
        location:
            Optical element, plane, or absolute longitudinal position z.
        occurrence:
            Occurrence of the element when the same object appears multiple times.

        Returns
        -------
        ParaxialState
            Propagated optical state at the requested location.
        """
        if isinstance(location, Real) and not isinstance(location, bool):
            if occurrence is not None:
                raise TypeError("occurrence may only be specified for optical elements.")

            if self._probe is None:
                raise ValueError("This simulation result does not support arbitrary-z probing.")

            return self._probe(location)

        before, _ = self._resolve_location(location, occurrence)

        return self.states[before]


    def after(self, element, occurrence=None) -> ParaxialState:
        """
        Return the optical state immediately after an element.

        Parameters
        ----------
        element:
            Element whose output state should be returned.
        occurrence:
            Occurrence of the element when the same object appears multiple times.

        Returns
        -------
        ParaxialState
            Propagated optical state immediately after the element.
        """
        _, after = self._resolve_location(element, occurrence)

        return self.states[after]

    def plot(self):
        """Plot this simulation result."""
        raise NotImplementedError