from __future__ import annotations

import numpy as np
from diffractix.backend import xp

from collections.abc import Sequence, Mapping
from functools import cached_property
from typing import TYPE_CHECKING

from diffractix.graph import InputNode, Parameter, Program
from .info import SimulationInput, SimulationStep
from .result import SimulationResult

if TYPE_CHECKING:
    from diffractix.system.info import ElementData, ParameterData
    from diffractix.solver.constraint import Constraint, ConstraintDescriptor



class InitialValues:
    """Editable theta values indexed by model Parameters or their handles."""

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
        return self.values[self._index(parameter)]

    def __setitem__(self, parameter: Parameter | InputNode, value):
        index = self._index(parameter)
        value = np.asarray(value)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError("An initial value must be a real numeric scalar.")
        if not np.isfinite(value):
            raise ValueError("An initial value must be finite.")
        self.values[index] = value



class Simulation:
    """
    Compiled differentiable optical simulation.

    A Simulation contains the numerical program and metadata required to
    evaluate an optical system. Running a simulation does not mutate the source
    System, its elements, or parameter graph.

    Programs capture graph expressions. Deferred callable constraints retain
    normal Python closure semantics.
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
        optical_inputs: Sequence[SimulationInput]
    ):
        self.element_data = tuple(element_data)
        self.parameter_data = tuple(parameter_data)
        self.callable_constraints = tuple(callable_constraints)
        self.constraint_descriptors = tuple(constraint_descriptors)
        self.constraint_program = constraint_program
        self.optics_program = optics_program
        self.optical_stages = tuple(optical_stages)
        self.optical_inputs = tuple(optical_inputs)

    @cached_property
    def _initial_values(self):
        """Initial theta in canonical parameter order."""
        values = np.array([parameter.value for parameter in self.parameter_data], dtype=float)
        values.setflags(write=False)
        return values

    @property
    def initial_values(self) -> InitialValues:
        """Return an independent editable copy of the build-time initial theta."""
        return InitialValues(self._initial_values, self.optics_program.parameter_indices)

    def run(self, theta=None) -> SimulationResult:
        """Evaluate the optics program once and propagate the numerical input."""
        if len(self.optical_inputs) != 1:
            raise ValueError("Simulation.run() requires exactly one optical input.")
        if theta is None:
            theta = self._initial_values

        values = self.optics_program.evaluate(theta)
        source = self.optical_inputs[0].construct(values)
        state = source
        z = 0.0

        states = [state]
        positions = [z]

        for step in self.optical_stages:
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
            optical_stages=self.optical_stages,
            optical_values=values,
            z=xp.stack(positions),
            states=tuple(states),
        )

    def __str__(self) -> str:
        def render_table(headers, rows):
            widths = [
                max([len(header), *(len(row[i]) for row in rows)])
                for i, header in enumerate(headers)
            ]

            def format_row(row):
                return "    ".join(value.ljust(width) for value, width in zip(row, widths))

            divider = "-" * (sum(widths) + 4 * (len(widths) - 1))
            return [format_row(headers), divider, *(format_row(row) for row in rows)]

        values = self.optics_program.evaluate(self._initial_values)
        has_paths = any(element.path for element in self.element_data)
        headers = ["#", "z [m]", "Type", "Label", *(["Path"] if has_paths else []), "L [m]", "n"]
        rows = []
        z = 0.0

        for element, stage in zip(self.element_data, self.optical_stages):
            length = values[stage.length_index]
            n = values[stage.refractive_index_index]
            rows.append((
                str(element.position),
                f"{z:.4g}",
                element.type_name,
                element.label or "-",
                *([element.path or "-"] if has_paths else []),
                f"{length:.4g}",
                f"{n:.4g}",
            ))
            z = z + length

        constraints = [
            (info.position, info.label or info.description)
            for info in self.constraint_descriptors
        ]
        constraints.extend(
            (position, constraint.label or str(constraint))
            for position, constraint in self.callable_constraints
        )
        constraints.sort(key=lambda item: item[0])
        sources = ", ".join(type(item.initial_state).__name__ for item in self.optical_inputs) or "-"

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
