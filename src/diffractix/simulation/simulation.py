from __future__ import annotations

import autograd.numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from diffractix.beams.base import ParaxialState
from diffractix.graph import CompiledAST, Node
from diffractix.simulation.result import SimulationResult, result_type_for

from diffractix.system.info import ElementInfo


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
        parameter_info: Mapping[int, ParameterInfo],
        compile_context: Mapping,
        execution_context: Mapping,
        location_map: Mapping,
        requirements: Sequence[Callable | Node],
        parameter_graph: CompiledAST,
        element_info: Sequence[ElementInfo],
    ):
        self.source = source
        self.graph = graph
        self.steps = tuple(steps)
        self.parameter_info = parameter_info
        self.location_map = location_map
        self.requirements = tuple(requirements)
        self.compile_context = compile_context
        self.parameter_graph = parameter_graph
        self.element_info = tuple(element_info)
        self.execution_context = execution_context

        self._result_type = result_type_for(self.source)

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
            Numerical trace of the propagated optical state.
        """
        if theta is None:
            theta = self.initial_values

        values = self.graph.evaluate(variable_values=theta, bindings=self.execution_context)

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
            element_info=self.element_info,
        )


    def __str__(self) -> str:
        col_gap = 4

        def format_value(v):
            return f"{float(v):.4g}"

        def format_bound(b, is_upper=False):
            if (np.isposinf(b) if is_upper else np.isneginf(b)):
                return "inf" if is_upper else "-inf"
            return format_value(b)

        def format_parameters(info, vals):
            if not info.parameter_names:
                return "-"
            return ", ".join(
                f"{name}={format_value(vals[idx])}"
                for name, idx in zip(info.parameter_names, info.parameter_indices)
            )

        def render_table(headers, rows):
            widths = [
                max(len(h), *(len(r[c]) for r in rows)) if rows else len(h)
                for c, h in enumerate(headers)
            ]
            fmt = lambda r: (" " * col_gap).join(val.ljust(w) for val, w in zip(r, widths))
            div = "-" * (sum(widths) + col_gap * (len(widths) - 1))
            return [fmt(headers), div, *(fmt(r) for r in rows)]

        values = self.graph.evaluate(self.initial_values, bindings=self.execution_context)
        values = self.graph.evaluate(
            self.initial_values,
            bindings=self.execution_context,
        )
        has_paths = any(info.path for info in self.element_info)
        headers = ["#", "z [m]", "Type", "Label", *(["Path"] if has_paths else []), "L [m]", "n", "Parameters"]

        rows = []
        z = 0.0
        for idx, (step, info) in enumerate(zip(self.steps, self.element_info)):
            length = values[step.length_index]
            n = values[step.refractive_index_index]
            rows.append((
                str(idx),
                format_value(z),
                info.type_name,
                info.label or "-",
                *([info.path or "-"] if has_paths else []),
                format_value(length),
                format_value(n),
                format_parameters(info, parameter_values),
            ))
            z += length

        variables = sorted(
            (info for info in self.parameter_info.values() if info.is_variable),
            key=lambda info: info.parameter_index,
        )

        lines = [
            "Compiled Simulation",
            "",
            *render_table(headers, rows),
            render_table(headers, rows)[1],  # bottom divider
            "",
            f"Source: {type(self.source).__name__}",
            f"Total length: {format_value(z)} m",
            f"Variables: {len(variables)}",
            f"Requirements: {len(self.requirements)}",
        ]

        if variables:
            var_headers = ("#", "Parameter", "Owner", "Initial", "Bounds")
            var_rows = [
                (
                    str(info.parameter_index),
                    info.name or "-",
                    info.owner_label or info.owner_type or "-",
                    format_value(info.value),
                    f"[{format_bound(info.lower_bound)}, {format_bound(info.upper_bound, is_upper=True)}]",
                )
                for info in variables
            ]
            lines.extend(["", "Variables", "", *render_table(var_headers, var_rows)])

        if self.requirements:
            lines.extend([
                "",
                "Requirements",
                "",
                *(f"{i}  {req}" for i, req in enumerate(self.requirements)),
            ])

        return "\n".join(lines)
