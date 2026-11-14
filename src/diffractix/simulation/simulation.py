from __future__ import annotations

import numpy as np
from diffractix.backend import xp

from collections.abc import Sequence, Mapping
from functools import cached_property
from typing import TYPE_CHECKING
from types import MappingProxyType

from diffractix.graph import InputNode, Parameter, Program
from .info import SimulationInput, SimulationStep
from .result import SimulationResult, SimulationResultTracer

if TYPE_CHECKING:
    from diffractix.system.info import ElementData, ParameterData
    from diffractix.solver.constraint import Constraint, ConstraintDescriptor



class InitialValues:
    """Editable copy of theta, the variable Parameters' values, indexed by Parameter.

    Returned by `Simulation.initial_values`; see `Simulation` for theta. It lets
    you read or set a value by its Parameter, as in ``guess[lens.f] = 0.2``,
    instead of by its position in theta. Its `values` attribute holds the plain
    array, to pass to :meth:`Simulation.run`. Editing it does not change the
    Simulation or the Parameters it was built from.

    Examples
    --------
    >>> from diffractix import Parameter, Ray, Space, System
    >>> space = Space(d=Parameter(0.05).variable())
    >>> simulation = System(space, optical_input=Ray(x=0, theta=0)).build()
    >>> guess = simulation.initial_values
    >>> guess[space.d] = 0.08
    >>> float(simulation.run(theta=guess.values)[space].z_end)
    0.08
    """

    def __init__(self, values: Sequence[float], parameter_indices: Mapping[int, int]):
        self.values = np.array(values, dtype=float, copy=True)
        self._parameter_indices = dict(parameter_indices)

    def _index(self, parameter: Parameter | InputNode) -> int:
        if isinstance(parameter, InputNode):
            parameter = parameter.node
        if not isinstance(parameter, Parameter):
            raise TypeError("Initial values can only be indexed by a Parameter or its direct InputNode handle.")
        if id(parameter) not in self._parameter_indices:
            raise KeyError(f"Parameter {parameter!r} is not part of this Simulation's theta.")
        return self._parameter_indices[id(parameter)]

    def __getitem__(self, parameter: Parameter | InputNode):
        """Return the current value of a variable Parameter.

        Parameters
        ----------
        parameter : Parameter or InputNode
            A variable Parameter in this theta, or its direct handle (for example
            ``lens.f``, not an expression built from it).

        Returns
        -------
        float
            The current value.

        Raises
        ------
        TypeError
            If `parameter` is not a Parameter or one of its direct handles.
        KeyError
            If `parameter` is not part of this Simulation's theta.
        """
        return self.values[self._index(parameter)]

    def __setitem__(self, parameter: Parameter | InputNode, value):
        """Set the value of a variable Parameter.

        Parameters
        ----------
        parameter : Parameter or InputNode
            A variable Parameter in this theta, or its direct handle.
        value : float
            The new value.

        Raises
        ------
        TypeError
            If `parameter` is not a Parameter or one of its direct handles, or `value`
            is not a real numeric scalar.
        ValueError
            If `value` is not finite.
        KeyError
            If `parameter` is not part of this Simulation's theta.
        """
        index = self._index(parameter)
        value = np.asarray(value)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError("An initial value must be a real numeric scalar.")
        if not np.isfinite(value):
            raise ValueError("An initial value must be finite.")
        self.values[index] = value



class Simulation:
    """A compiled optical system, ready to run and optimize.

    :meth:`System.build` produces a Simulation. It is a snapshot: later changes
    to the System, its elements or their parameters do not reach it.

    Attributes
    ----------
    element_data : tuple of ElementData
        Resolved elements, generated gaps included, in propagation order.
    parameter_data : tuple of ParameterData
        Variable Parameters in theta order, with their names, initial values and
        bounds.
    bindings : Mapping
        Values bound for Symbols, by key, captured at :meth:`System.build`.

    Notes
    -----
    The values of the variable Parameters, in `parameter_data` order, form one
    1-D array called *theta*. :meth:`run` takes a theta, the Solver searches for
    the best one, and `initial_values` holds the build-time theta.

    See Also
    --------
    System.build : Produces a Simulation.
    """

    def __init__(
        self,
        *,
        element_data: Sequence[ElementData],
        parameter_data: Sequence[ParameterData],
        callable_constraints: Sequence[tuple[int, Constraint]],
        constraint_descriptors: Sequence[ConstraintDescriptor],
        constraint_program: Program,
        optics_program: Program,
        optical_stages: Sequence[SimulationStep],
        optical_inputs: Sequence[SimulationInput],
        bindings: Mapping = {},
    ):
        self.element_data = tuple(element_data)
        self.parameter_data = tuple(parameter_data)
        self._callable_constraints = tuple(callable_constraints)
        self._constraint_descriptors = tuple(constraint_descriptors)
        self._constraint_program = constraint_program
        self._optics_program = optics_program
        self._optical_stages = tuple(optical_stages)
        self._optical_inputs = tuple(optical_inputs)
        self.bindings = MappingProxyType(dict(bindings))

    @cached_property
    def symbolic(self):
        """Symbolic result: this Simulation's result as graph expressions, for
        writing Solver objectives and constraints.

        Its values depend on the variable Parameters instead of being fixed
        numbers. For example, ``simulation.symbolic[lens].after.w <= 1 * mm``
        is a constraint on the beam radius after `lens`, which the Solver checks
        each time it runs the simulation. Use it only with a Solver for this
        same Simulation.

        Returns
        -------
        SimulationResultTracer
            The symbolic result.

        Raises
        ------
        ValueError
            If the Simulation does not have exactly one optical input.
        """
        if len(self._optical_inputs) != 1:
            raise ValueError("Simulation.symbolic requires exactly one optical input.")
        return SimulationResultTracer(self.element_data, self._optical_inputs[0].initial_state)

    @cached_property
    def _initial_values(self):
        """Initial theta in canonical parameter order."""
        values = np.array([parameter.value for parameter in self.parameter_data], dtype=float)
        values.setflags(write=False)
        return values

    @property
    def initial_values(self) -> InitialValues:
        """Initial values: the variable Parameters' build-time values (theta; see
        `Simulation`), as an editable copy.

        Edits do not change the Simulation or its Parameters; pass the edited
        `InitialValues.values` to :meth:`run`.

        Returns
        -------
        InitialValues
            The editable copy.
        """
        return InitialValues(self._initial_values, self._optics_program.parameter_indices)

    def run(self, theta=None) -> SimulationResult:
        """Run the simulation: propagate the optical source through every element.

        Parameters
        ----------
        theta : array_like, optional
            Values for the variable Parameters, in the order of `parameter_data`.
            Defaults to the build-time initial values.

        Returns
        -------
        SimulationResult
            The states and positions along the system, for this theta.

        Raises
        ------
        ValueError
            If the Simulation does not have exactly one optical input.

        Examples
        --------
        >>> from diffractix import Ray, Space, System, ThinLens
        >>> from diffractix.units import mm
        >>> system = System(ThinLens(f=100 * mm), Space(d=100 * mm), optical_input=Ray(x=1 * mm, theta=0))
        >>> result = system.build().run()
        >>> round(float(result.final.x), 9)
        0.0
        """
        if len(self._optical_inputs) != 1:
            raise ValueError("Simulation.run() requires exactly one optical input.")
        if theta is None:
            theta = self._initial_values

        values = self._optics_program.evaluate(theta)
        source = self._optical_inputs[0].construct(values)
        state = source
        z = 0.0

        states = [state]
        positions = [z]

        for step in self._optical_stages:
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

        return SimulationResult(
            theta=theta,
            parameter_data=self.parameter_data,
            element_data=self.element_data,
            optical_stages=self._optical_stages,
            optical_values=values,
            z=xp.stack(positions),
            states=tuple(states),
        )

    def __str__(self) -> str:
        """Return a readable summary of the simulation (compiled system) and its variables and constraints."""
        def render_table(headers, rows):
            widths = [
                max([len(header), *(len(row[i]) for row in rows)])
                for i, header in enumerate(headers)
            ]

            def format_row(row):
                return "    ".join(value.ljust(width) for value, width in zip(row, widths))

            divider = "-" * (sum(widths) + 4 * (len(widths) - 1))
            return [format_row(headers), divider, *(format_row(row) for row in rows)]

        values = self._optics_program.evaluate(self._initial_values)
        has_paths = any(element.path for element in self.element_data)
        headers = ["#", "z [m]", "Type", "Label", *(["Path"] if has_paths else []), "L [m]", "n"]
        rows = []
        z = 0.0

        for element, stage in zip(self.element_data, self._optical_stages):
            length = values[stage.length_index]
            n = values[stage.refractive_index_index]
            rows.append((
                str(element.position),
                f"{z:.4g}",
                element.type_name,
                element.label or "-",
                *([".".join(element.path) if element.path else "-"] if has_paths else []),
                f"{length:.4g}",
                f"{n:.4g}",
            ))
            z = z + length

        constraints = [
            (info.position, info.label or info.description)
            for info in self._constraint_descriptors
        ]
        constraints.extend(
            (position, constraint.label or str(constraint))
            for position, constraint in self._callable_constraints
        )
        constraints.sort(key=lambda item: item[0])
        sources = ", ".join(type(item.initial_state).__name__ for item in self._optical_inputs) or "-"

        lines = [
            "Compiled Simulation",
            "",
            *render_table(headers, rows),
            "",
            f"Source: {sources}",
            f"Total length: {z:.4g} m",
            f"Variables: {len(self.parameter_data)}",
            f"Constraints: {len(constraints)}",
        ]

        if self.parameter_data:
            variable_rows = [
                (
                    str(parameter.parameter_index),
                    parameter.name or "-",
                    f"{parameter.value:.4g}",
                    f"[{parameter.lower_bound:.4g}, {parameter.upper_bound:.4g}]",
                )
                for parameter in self.parameter_data
            ]
            lines.extend(["", "Variables", "", *render_table(("#", "Parameter", "Initial", "Bounds"), variable_rows)])

        if constraints:
            lines.extend(["", "Constraints", "", *(f"{position}  {description}" for position, description in constraints)])

        return "\n".join(lines)
