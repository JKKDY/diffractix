from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from functools import cache
from numbers import Integral, Real

import autograd.numpy as np

from diffractix.beams.base import ParaxialState
from diffractix.system.info import ElementInfo


class SimulationResult:
    """Numerical result produced by a Simulation."""

    def __init__(
        self,
        source: ParaxialState,
        z: np.ndarray,
        states: Sequence[ParaxialState],
        location_map: Mapping[int, tuple[tuple[int, int], ...]],
        element_info: Sequence[ElementInfo],
        probe: Callable[[float], ParaxialState] | None = None,
    ):
        if len(z) != len(states):
            raise ValueError(
                f"z and states must have the same length, got {len(z)} and {len(states)}."
            )
        if not states:
            raise ValueError("SimulationResult requires at least one state.")
        if len(states) != len(element_info) + 1:
            raise ValueError(
                "states must contain exactly one input state plus one state "
                f"per element_info entry, got {len(states)} states and "
                f"{len(element_info)} element_info entries."
            )
        self._source = source
        self.z = z
        self.states = tuple(states)
        self._location_map = dict(location_map)
        self.element_info = tuple(element_info)
        self._probe = probe

    @property
    def source(self) -> ParaxialState:
        """Input optical state used to create this simulation result."""
        return self._source

    @property
    def initial(self) -> ParaxialState:
        """Optical state at the beginning of the simulation."""
        return self.states[0]

    @property
    def final(self) -> ParaxialState:
        """Optical state at the end of the simulation."""
        return self.states[-1]

    def _resolve_location(self, element, occurrence=None):
        try:
            locations = self._location_map[id(element)]
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
            return locations[0]
        if isinstance(occurrence, bool) or not isinstance(occurrence, Integral):
            raise TypeError(
                f"occurrence must be an integer, got {type(occurrence).__name__}."
            )
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
                raise TypeError(
                    "occurrence may only be specified for optical elements."
                )
            if self._probe is None:
                raise ValueError(
                    "This simulation result does not support arbitrary-z probing."
                )
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

    def z_at(self, element, occurrence=None):
        """Return the longitudinal position at an element's input side."""
        before, _ = self._resolve_location(element, occurrence)
        return self.z[before]

    def z_after(self, element, occurrence=None):
        """Return the longitudinal position immediately after an element."""
        _, after = self._resolve_location(element, occurrence)
        return self.z[after]

    def plot(self):
        """Plot this simulation result."""
        raise NotImplementedError

    def __str__(self) -> str:
        def fmt_val(v):
            if  np.ndim(v) == 0:
                return f"{float(v):.4g}"
            else:
                return np.array2string(np.asarray(v), precision=4, separator=", ")

        def fmt_loc(info):
            return info.label or info.path or info.type_name

        state_type = type(self.source)
        cols = state_type.result_columns
        headers = ("#", "z", "Location", *cols)

        # Consolidate row generation across input (0) and subsequent elements
        data_items = [(0, self.z[0], self.states[0], "Input")] + [
            (i, z, state, fmt_loc(info))
            for i, (z, state, info) in enumerate(zip(self.z[1:], self.states[1:], self.element_info), start=1)
        ]
        rows = [
            (str(i), fmt_val(z), loc, *(fmt_val(getattr(state, c)) for c in cols))
            for i, z, state, loc in data_items
        ]

        widths = [max(len(header), *(len(r[i]) for r in rows)) for i, header in enumerate(headers)]
        fmt_row = lambda r: "    ".join(val.ljust(w) for val, w in zip(r, widths))
        divider = "-" * (sum(widths) + 4 * (len(widths) - 1))

        return "\n".join([
            f"{state_type.__name__} Simulation Result",
            "",
            fmt_row(headers),
            divider,
            *(fmt_row(r) for r in rows),
            divider,
            "",
            f"Final z: {fmt_val(self.z[-1])}",
            f"States: {len(self.states)}",
        ])


def _state_property(name):
    """Create a result property that stacks one state attribute over the trace."""

    def getter(self):
        return np.stack([
            getattr(state, name)
            for state in self.states
        ])

    getter.__name__ = name
    getter.__doc__ = (
        f"Values of `{name}` for every propagated optical state."
    )

    return property(getter)



def result_type_for(source: ParaxialState) -> type[SimulationResult]:
    """
    Return the generated SimulationResult type for an optical state.

    Public dataclass fields and properties of the state become stacked
    properties on the generated result type.
    """
    return _result_type_for(type(source))



@cache
def _result_type_for(state_type: type[ParaxialState]) -> type[SimulationResult]:
    """Generate and cache a SimulationResult type for a ParaxialState type."""

    if not is_dataclass(state_type):
        raise TypeError(
            f"{state_type.__name__} must be a dataclass to generate a simulation result type."
        )

    names = {
        field.name
        for field in fields(state_type)
        if not field.name.startswith("_")
    }

    for cls in state_type.__mro__:
        names.update(
            name
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, property)
        )

    collisions = names.intersection(dir(SimulationResult))

    if collisions:
        raise TypeError(
            f"{state_type.__name__} exposes result properties that conflict with "
            f"SimulationResult: {', '.join(sorted(collisions))}."
        )

    properties = {
        name: _state_property(name)
        for name in names
    }

    return type(
        f"{state_type.__name__}SimulationResult",
        (SimulationResult,),
        properties,
    )
