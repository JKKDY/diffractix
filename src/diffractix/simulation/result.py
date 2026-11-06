from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from diffractix.backend import xp

from diffractix.input.paraxial_input import ParaxialState
if TYPE_CHECKING:
    from diffractix.system.info import ElementData, ParameterData
    from .info import SimulationStep


class SimulationResult:
    """Numerical result produced by a Simulation."""

    class Stage:
        """Boundary states and positions for one resolved optical element."""

        def __init__(self, result: SimulationResult, index: int):
            self._result = result
            self._index = index

        @property
        def before(self) -> ParaxialState:
            return self._result.states[self._index]

        @property
        def after(self) -> ParaxialState:
            return self._result.states[self._index + 1]

        @property
        def z_start(self):
            return self._result.z[self._index]

        @property
        def z_end(self):
            return self._result.z[self._index + 1]

    def __init__(
        self,
        *,
        theta: np.ndarray,
        parameter_data: Sequence[ParameterData],
        element_data: Sequence[ElementData],
        optical_stages: Sequence[SimulationStep],
        optical_values: Sequence,
        z: np.ndarray,
        states: Sequence[ParaxialState],
    ):
        self.theta = xp.array(theta)
        self.parameter_data = tuple(parameter_data)
        self.element_data = tuple(element_data)
        self.optical_stages = tuple(optical_stages)
        self.optical_values = tuple(optical_values)
        self.z = z
        self.states = tuple(states)

        if len(z) != len(states):
            raise ValueError(
                f"z and states must have the same length, got {len(z)} and {len(states)}."
            )
        if not self.states:
            raise ValueError("SimulationResult requires at least one state.")
        if len(self.states) != len(self.element_data) + 1:
            raise ValueError(
                "states must contain exactly one input state plus one state "
                f"per element_data entry, got {len(self.states)} states and "
                f"{len(self.element_data)} element_data entries."
            )
        if len(self.optical_stages) != len(self.element_data):
            raise ValueError("optical_stages and element_data must have the same length.")
        if len(self.theta) != len(self.parameter_data):
            raise ValueError("theta and parameter_data must have the same length.")

        self._element_positions = {}
        for index, element in enumerate(self.element_data):
            if element.element_id not in self._element_positions:
                self._element_positions[element.element_id] = []
            self._element_positions[element.element_id].append(index)

    @property
    def source(self) -> ParaxialState:
        """Input optical state used to create this simulation result."""
        return self.states[0]

    @property
    def initial(self) -> ParaxialState:
        """Optical state at the beginning of the simulation."""
        return self.states[0]

    @property
    def final(self) -> ParaxialState:
        """Optical state at the end of the simulation."""
        return self.states[-1]

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Stage views in propagation order."""
        return tuple(self.Stage(self, index) for index in range(len(self.element_data)))

    def stage(self, element, occurrence=None) -> Stage:
        """Return a stage view, selecting an occurrence for repeated elements."""
        before, _ = self._resolve_location(element, occurrence)
        return self.Stage(self, before)

    def __getitem__(self, key) -> Stage:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Expected result[element] or result[element, occurrence].")
            element, occurrence = key
            return self.stage(element, occurrence=occurrence)
        return self.stage(key)

    def _resolve_location(self, element, occurrence=None):
        try:
            locations = self._element_positions[id(element)]
        except KeyError:
            raise KeyError(
                f"Element {element!r} is not part of this simulation."
            ) from None
        if occurrence is None:
            if len(locations) > 1:
                raise ValueError(
                    f"Element {element!r} occurs {len(locations)} times in this simulation. "
                    "Specify occurrence= to select one."
                )
            index = locations[0]
            return index, index + 1
        if isinstance(occurrence, bool) or not isinstance(occurrence, Integral):
            raise TypeError(
                f"occurrence must be an integer, got {type(occurrence).__name__}."
            )
        if occurrence < 0 or occurrence >= len(locations):
            raise IndexError(
                f"Element {element!r} has {len(locations)} occurrence(s); "
                f"occurrence {occurrence} is out of range."
            )
        index = locations[occurrence]
        return index, index + 1

    def at(self, location, occurrence=None) -> ParaxialState:
        """
        Return the optical state at a location.

        Parameters
        ----------
        location:
            Optical element participating in the simulation.
        occurrence:
            Occurrence of the element when the same object appears multiple times.

        Returns
        -------
        ParaxialState
            Propagated optical state at the requested location.
        """
        if isinstance(location, Real):
            if occurrence is not None:
                raise TypeError("occurrence may only be specified when querying an element.")

            raise ValueError("This simulation result does not support arbitrary-z probing.")

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

    def z_at(self, element, occurrence=None):
        """Return the longitudinal position at an element's input side."""
        before, _ = self._resolve_location(element, occurrence)
        return self.z[before]

    def z_after(self, element, occurrence=None):
        """Return the longitudinal position immediately after an element."""
        _, after = self._resolve_location(element, occurrence)
        return self.z[after]

    def plot(self, ax=None, *, show_elements=True, show_axis=True):
        """Plot a concrete 2D trace, sampling only known homogeneous Spaces.

        Returns ``(fig, ax)`` without showing or saving the figure.
        Lengths are interpreted as metres. Matplotlib is included with Diffractix.
        """
        from diffractix.plotting import plot

        return plot(self, ax=ax, show_elements=show_elements, show_axis=show_axis)

    def __str__(self) -> str:
        def format_value(value):
            if np.ndim(value) == 0:
                return f"{value:.4g}"
            return np.array2string(np.asarray(value), precision=4, separator=", ")

        state_type = type(self.source)
        cols = state_type.result_columns
        headers = ("#", "z", "Location", *cols)

        rows = []
        for index, (z, state) in enumerate(zip(self.z, self.states)):
            if index == 0:
                location = "Input"
            else:
                element = self.element_data[index - 1]
                location = element.label or element.path or element.type_name
            rows.append((
                str(index), format_value(z), location,
                *(format_value(getattr(state, column)) for column in cols),
            ))

        widths = [max(len(header), *(len(r[i]) for r in rows)) for i, header in enumerate(headers)]
        def format_row(row):
            return "    ".join(value.ljust(width) for value, width in zip(row, widths))

        divider = "-" * (sum(widths) + 4 * (len(widths) - 1))

        return "\n".join([
            f"{state_type.__name__} Simulation Result",
            "",
            format_row(headers),
            divider,
            *(format_row(row) for row in rows),
            divider,
            "",
            f"Final z: {format_value(self.z[-1])}",
            f"States: {len(self.states)}",
        ])
