from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from inspect import getattr_static
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from diffractix.backend import xp

from diffractix.states.paraxial_state import ParaxialState
from diffractix.graph import Symbol, SymbolicControlFlowError
if TYPE_CHECKING:
    from diffractix.system.info import ElementData, ParameterData
    from .info import SimulationStep





class SimulationResult:
    """Numerical optical states and positions produced by running a `Simulation`.

    Produced by :meth:`Simulation.run`. Index it with a resolved element to get the
    states and positions at its boundaries.

    Attributes
    ----------
    theta : array_like
        The variable Parameters' values used for this run, in `parameter_data`
        order; see `Simulation`.
    parameter_data : tuple of ParameterData
        Variable Parameters in theta order, with their names and bounds. Their
        `value` fields are the build-time defaults, not this run's values; read
        those from `theta`.
    element_data : tuple of ElementData
        Resolved elements, generated gaps included, in propagation order.
    z : array_like
        Position [m] at the input and after every element.
    states : tuple of ParaxialState
        Optical state at the input and after every element.

    Raises
    ------
    ValueError
        If `z` and `states` have different lengths, `states` is empty, `states`
        does not hold exactly one entry per `element_data` entry plus the input,
        `optical_stages` and `element_data` have different lengths, or `theta`
        and `parameter_data` have different lengths.
    """

    class Stage:
        """One element's stage: the optical states entering and leaving it, and
        where it starts and ends."""

        def __init__(self, result: SimulationResult, index: int):
            self._result = result
            self._index = index

        @property
        def before(self) -> ParaxialState:
            """Before: the optical state entering the element."""
            return self._result.states[self._index]

        @property
        def after(self) -> ParaxialState:
            """After: the optical state leaving the element."""
            return self._result.states[self._index + 1]

        @property
        def z_start(self):
            """Position [m]: where the element starts."""
            return self._result.z[self._index]

        @property
        def z_end(self):
            """Position [m]: where the element ends."""
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
        self._optical_stages = tuple(optical_stages)
        self._optical_values = tuple(optical_values)
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
        if len(self._optical_stages) != len(self.element_data):
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
        """Source: the optical state at the beginning of the simulation. Same as `initial`."""
        return self.states[0]

    @property
    def initial(self) -> ParaxialState:
        """Initial: the optical state at the beginning of the simulation."""
        return self.states[0]

    @property
    def final(self) -> ParaxialState:
        """Final: the optical state at the end of the simulation."""
        return self.states[-1]

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Stages: one per resolved element, generated gaps included, in propagation order."""
        return tuple(self.Stage(self, index) for index in range(len(self.element_data)))

    def view(self) -> SimulationResultView:
        """Return a read-only view of this result, for use in callable Solver objectives and constraints.

        Returns
        -------
        SimulationResultView
            The view.
        """
        return SimulationResultView(self)

    def stage(self, element, occurrence=None) -> Stage:
        """Return a stage for a resolved element, selecting an occurrence if it repeats.

        Parameters
        ----------
        element : OpticalElement
            An element of the built system. Composites are resolved to their leaf
            elements, so look up the leaf, for example ``relay.lens1``.
        occurrence : int, optional
            Which occurrence to select when `element` appears more than once.
            Required in that case.

        Returns
        -------
        SimulationResult.Stage
            The stage.

        Raises
        ------
        KeyError
            If `element` is not part of this simulation.
        TypeError
            If `occurrence` is given and is not an integer.
        IndexError
            If `occurrence` is out of range.
        ValueError
            If `element` occurs more than once and `occurrence` is not given.
        """
        return self.Stage(self, _stage_index(self._element_positions, element, occurrence))

    def __getitem__(self, key) -> Stage:
        """Return a stage for a resolved element; see :meth:`stage`.

        Notes
        -----
        Index with ``result[element]``, or ``result[element, occurrence]`` for a
        repeated element.

        Examples
        --------
        >>> from diffractix import Ray, Space, System, ThinLens
        >>> lens = ThinLens(f=0.1)
        >>> result = System(lens, optical_input=Ray(x=1e-3, theta=0)).build().run()
        >>> float(result[lens].before.x), float(result[lens].after.x)
        (0.001, 0.001)
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Expected result[element] or result[element, occurrence].")
            element, occurrence = key
            return self.stage(element, occurrence=occurrence)
        return self.stage(key)

    def plot(self, ax=None, *, show_elements=True, show_axis=True):
        """Plot a concrete 2D trace of this result.

        Only known homogeneous Spaces are sampled between boundaries; see Notes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Defaults to ``None``: a new figure and axes are created.
        show_elements : bool, optional
            Whether to draw and label the elements. Defaults to ``True``.
        show_axis : bool, optional
            Whether to draw the optical axis. Defaults to ``True``.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Neither is shown or saved; call ``fig.savefig(...)`` or ``plt.show()``
            yourself.

        Raises
        ------
        ImportError
            If Matplotlib cannot be imported.

        Notes
        -----
        Supports `Ray`, `RayBundle` and `GaussianBeam` results; a bundle shows up
        to 41 rays. A thick element such as `GRIN` retains only its boundary
        states; the interior is not sampled. General probing at an arbitrary z
        is not supported.
        """
        from diffractix.plotting import plot

        return plot(self, ax=ax, show_elements=show_elements, show_axis=show_axis)

    def __str__(self) -> str:
        """Return a readable summary of the states and positions along the trace."""
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
                location = element.label or ".".join(element.path or ()) or element.type_name
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



@dataclass(frozen=True)
class SimulationResultView:
    """Read-only view of a `SimulationResult`, passed to callable Solver objectives and constraints.

    Returned by :meth:`SimulationResult.view`. A callable objective or constraint
    receives one when its quantity cannot be written with `Simulation.symbolic`,
    for example because it uses an ``if`` on the result's values. The view gives
    the result's own state objects, not copies.

    See Also
    --------
    SimulationResult.view : Returns a view of a result.
    Simulation.symbolic : The graph-expression counterpart, for quantities that
        don't need Python control flow.
    """

    _result: SimulationResult

    @property
    def initial(self) -> ParaxialState:
        """Initial: the optical state at the beginning of the simulation."""
        return self._result.initial

    @property
    def final(self) -> ParaxialState:
        """Final: the optical state at the end of the simulation."""
        return self._result.final

    @property
    def stages(self) -> tuple[SimulationResult.Stage, ...]:
        """Stages: one per resolved element, generated gaps included, in propagation order."""
        return self._result.stages

    def __getitem__(self, key) -> SimulationResult.Stage:
        """Return a stage for a resolved element; see :meth:`SimulationResult.__getitem__`."""
        return self._result[key]





class ResultSymbolKind(Enum):
    """Which part of a SimulationResult a result Symbol stands for."""
    INITIAL = auto()    # a field of the input state
    FINAL = auto()      # a field of the last state
    BEFORE = auto()     # a field of the state entering stage `index`
    AFTER = auto()      # a field of the state leaving stage `index`
    Z_START = auto()    # where stage `index` starts
    Z_END = auto()      # where stage `index` ends


@dataclass(frozen=True)
class ResultSymbolKey:
    """Key of a Symbol produced by SimulationResultTracer; resolved against
    each run's SimulationResult by `resolve_symbol`.

    `index` is the stage index for stage kinds; `name` is the state field for
    state kinds.
    """
    kind: ResultSymbolKind
    index: int | None = None
    name: str | None = None


def resolve_symbol(key, result):
    """Return the value a result Symbol's key stands for in a SimulationResult."""
    if not isinstance(key, ResultSymbolKey):
        raise KeyError(f"Unsupported runtime Symbol: {key!r}.")

    if key.kind is ResultSymbolKind.INITIAL:
        return getattr(result.initial, key.name)
    if key.kind is ResultSymbolKind.FINAL:
        return getattr(result.final, key.name)
    if key.kind is ResultSymbolKind.BEFORE:
        return getattr(result.states[key.index], key.name)
    if key.kind is ResultSymbolKind.AFTER:
        return getattr(result.states[key.index + 1], key.name)
    if key.kind is ResultSymbolKind.Z_START:
        return result.z[key.index]
    if key.kind is ResultSymbolKind.Z_END:
        return result.z[key.index + 1]

    raise KeyError(f"Unknown result Symbol: {key!r}.")


class _ResultSymbol(Symbol):
    def __getitem__(self, index):
        raise SymbolicControlFlowError("Indexing symbolic result values requires runtime evaluation.")


def _stage_index(positions, element, occurrence):
    """Return the stage index of `element`, or of its chosen occurrence.

    `positions` maps ``id(element)`` to that element's stage indices, in
    propagation order; an element repeated in the system has several.
    """
    try:
        locations = positions[id(element)]
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


class SimulationResultTracer:
    """Symbolic `SimulationResult`, for writing Solver objectives and constraints.

    Returned by `Simulation.symbolic`. It has the same interface as
    `SimulationResult`: `initial`, `final`, `stages` and indexing by element.
    But every value is a graph expression instead of a number. The Solver
    evaluates these expressions each time it runs the simulation with new
    values for the variable Parameters.

    Examples
    --------
    Find the distance after a lens where a ray crosses the axis:

    >>> from diffractix import Parameter, Ray, Solver, Space, System, ThinLens
    >>> space = Space(d=Parameter(0.05).variable())
    >>> system = System(ThinLens(f=0.1), space, optical_input=Ray(x=1e-3, theta=0))
    >>> simulation = system.build()
    >>> sim = simulation.symbolic
    >>> solver = Solver(simulation).target(value=sim.final.x, target=0.0, scale=1e-3)
    >>> solution = solver.solve()
    >>> round(float(solution.value(space.d)), 6)
    0.1

    See Also
    --------
    SimulationResult : The numerical result this mirrors.
    Simulation.symbolic : Returns a tracer bound to a Simulation.
    """

    class SymbolicState:
        """Symbolic state: the light at one point of the system, with each field
        as a graph expression.

        Produced by the tracer's `initial` and `final`, and by a stage's `before`
        and `after`. It holds no values: reading a field, such as ``x`` or ``w``,
        produces a Symbol that the Solver fills in when it runs the simulation.

        Raises
        ------
        AttributeError
            If the state has no such attribute.
        SymbolicControlFlowError
            If the attribute is a method, such as `Ray.x_at_z`, or a dynamic
            attribute computed from the backend's own control flow; call it on
            the numerical result instead, or use a callable constraint.
        """
        def __init__(self, state, kind, index=None):
            self._state = state
            self._kind = kind
            self._index = index

        def __getattr__(self, name):
            state_type = type(self._state).__name__
            if name.startswith("_"):
                raise AttributeError(f"Private state attribute {state_type}.{name} cannot be traced.")
            try:
                attribute = getattr_static(self._state, name)
            except AttributeError:
                if getattr_static(type(self._state), "__getattr__", None) is not None:
                    raise SymbolicControlFlowError(
                        f"Dynamic state attribute {state_type}.{name} requires runtime evaluation."
                    ) from None
                raise AttributeError(f"State {state_type} has no attribute {name!r}.") from None
            if callable(attribute):
                raise SymbolicControlFlowError(
                    f"State method {state_type}.{name} requires runtime evaluation."
                )
            return _ResultSymbol(ResultSymbolKey(self._kind, self._index, name))

        def __bool__(self):
            raise SymbolicControlFlowError("Symbolic values cannot be used as Python booleans.")

        def __eq__(self, other):
            raise SymbolicControlFlowError("Symbolic states cannot be compared in Python control flow.")

        def __ne__(self, other):
            raise SymbolicControlFlowError("Symbolic states cannot be compared in Python control flow.")

    class OpticalStage:
        """Symbolic stage: one element's passage, that is the light entering and
        leaving it, and where it starts and ends.

        Produced by indexing the tracer, as in ``sim[lens]``, or by its `stages`.
        `before` and `after` give a `SymbolicState` (a whole state); `z_start`
        and `z_end` give graph expressions. Mirrors `SimulationResult.Stage`.
        """
        def __init__(self, state, index):
            self._state = state
            self._index = index

        @property
        def before(self):
            """Before: the optical state entering the element, as a symbolic state."""
            return SimulationResultTracer.SymbolicState(self._state, ResultSymbolKind.BEFORE, self._index)

        @property
        def after(self):
            """After: the optical state leaving the element, as a symbolic state."""
            return SimulationResultTracer.SymbolicState(self._state, ResultSymbolKind.AFTER, self._index)

        @property
        def z_start(self):
            """Position [m]: where the element starts, as a graph expression."""
            return Symbol(ResultSymbolKey(ResultSymbolKind.Z_START, self._index))

        @property
        def z_end(self):
            """Position [m]: where the element ends, as a graph expression."""
            return Symbol(ResultSymbolKey(ResultSymbolKind.Z_END, self._index))

        def __bool__(self):
            raise SymbolicControlFlowError("Symbolic values cannot be used as Python booleans.")

        def __eq__(self, other):
            raise SymbolicControlFlowError("Symbolic stages cannot be compared in Python control flow.")

        def __ne__(self, other):
            raise SymbolicControlFlowError("Symbolic stages cannot be compared in Python control flow.")

    def __init__(self, element_data: Sequence[ElementData], state: ParaxialState):
        self._state = state
        self._stage_count = len(element_data)
        self._element_positions = {}
        for index, element in enumerate(element_data):
            self._element_positions.setdefault(element.element_id, []).append(index)

    @property
    def initial(self):
        """Initial: the optical state at the beginning of the simulation, as a symbolic state."""
        return self.SymbolicState(self._state, ResultSymbolKind.INITIAL)

    @property
    def final(self):
        """Final: the optical state at the end of the simulation, as a symbolic state."""
        return self.SymbolicState(self._state, ResultSymbolKind.FINAL)

    @property
    def stages(self):
        """Symbolic stages: one per resolved element, generated gaps included, in propagation order."""
        return tuple(self.OpticalStage(self._state, index) for index in range(self._stage_count))

    def __getitem__(self, key):
        """Return a symbolic stage for a resolved element.

        Parameters
        ----------
        element : OpticalElement
            An element of the built system. Composites are resolved to their leaf
            elements, so look up the leaf, for example ``relay.lens1``.
        occurrence : int, optional
            Which occurrence to select when `element` appears more than once.
            Required in that case.

        Returns
        -------
        SimulationResultTracer.OpticalStage
            The symbolic stage.

        Raises
        ------
        KeyError
            If `element` is not part of this simulation.
        TypeError
            If `occurrence` is given and is not an integer.
        IndexError
            If `occurrence` is out of range.
        ValueError
            If `element` occurs more than once and `occurrence` is not given.

        Notes
        -----
        Index with ``result[element]``, or ``result[element, occurrence]`` for a
        repeated element.
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Expected result[element] or result[element, occurrence].")
            element, occurrence = key
        else:
            element, occurrence = key, None
        return self.OpticalStage(self._state, _stage_index(self._element_positions, element, occurrence))



