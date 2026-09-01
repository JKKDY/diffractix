from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

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
        location_map: Mapping[object, tuple[int, int]],
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

    def at(self, location) -> ParaxialState:
        """
        Return the optical state at a location.

        Parameters
        ----------
        location:
            Optical element, plane, or absolute longitudinal position z.

        Returns
        -------
        ParaxialState
            Propagated optical state at the requested location.
        """
        if isinstance(location, Real) and not isinstance(location, bool):
            if self._probe is None:
                raise ValueError("This simulation result does not support arbitrary-z probing.")

            return self._probe(location)

        try:
            before, _ = self._location_map[location]
        except KeyError:
            raise KeyError(
                f"Location {location!r} is not part of this simulation."
            ) from None
        except TypeError:
            raise TypeError(
                f"Expected an optical location or numeric z position, got {type(location).__name__}."
            ) from None

        return self._states[before]

    def after(self, element) -> ParaxialState:
        """
        Return the optical state immediately after an element.

        Parameters
        ----------
        element:
            Element whose output state should be returned.

        Returns
        -------
        ParaxialState
            Propagated optical state immediately after the element.
        """
        try:
            _, after = self._location_map[element]
        except KeyError:
            raise KeyError(
                f"Element {element!r} is not part of this simulation."
            ) from None
        except TypeError:
            raise TypeError(
                f"Expected an optical element, got {type(element).__name__}."
            ) from None

        return self._states[after]

    def plot(self):
        """Plot this simulation result."""
        raise NotImplementedError