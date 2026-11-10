from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from numbers import Real
from typing import Any

import numpy as np

from diffractix.graph import InputNode, Node, Parameter, Symbol, evaluate_ast, walk_ast
from diffractix.simulation import Simulation
from diffractix.simulation.result import ResultSymbolKey, resolve_symbol
from diffractix.system.info import ParameterData

from .scale import Scaling


@dataclass(frozen=True)
class SolverResult:
    """Optimization result; Solver returns x in physical SI coordinates."""

    x: np.ndarray
    success: bool
    cost: float
    message: str
    # Iteration and objective-evaluation counts as the backend reports them;
    # None when a backend does not provide one.
    iterations: int | None = None
    function_evaluations: int | None = None
    raw: Any = None
    scaling: Scaling | None = None


DEFAULT_FEASIBILITY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ObjectiveResult:
    position: int
    label: str | None
    value: object
    weight: float
    cost: float


@dataclass(frozen=True)
class ConstraintResult:
    position: int
    label: str | None
    description: str
    value: object
    lower_bound: float
    upper_bound: float
    violation: object
    margin: object
    satisfied: bool


@dataclass(frozen=True)
class ParameterBoundResult:
    parameter: ParameterData
    value: float
    lower_bound: float
    upper_bound: float
    violation: float
    margin: float
    satisfied: bool


class Solution:
    """Solved values and complete model evaluations in physical coordinates."""

    def __init__(
        self, result: SolverResult, problem, simulation: Simulation, *,
        objective_program, objective_descriptors, callable_objectives,
        constraint_program, constraint_descriptors, callable_constraints,
        feasibility_tolerance=DEFAULT_FEASIBILITY_TOLERANCE,
    ):
        if isinstance(feasibility_tolerance, bool) or not isinstance(feasibility_tolerance, Real):
            raise TypeError("feasibility_tolerance must be a real numeric scalar.")
        if not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and non-negative.")

        x = np.array(result.x, copy=True)
        if x.shape != (len(simulation.parameter_data),):
            raise ValueError("Backend result x shape must match simulation.initial_values.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Backend result x must contain only finite values.")
        x.setflags(write=False)
        self._result = result
        self._x = x
        self._feasibility_tolerance = feasibility_tolerance
        self._objective_program = objective_program
        self._objective_descriptors = tuple(objective_descriptors)
        self._callable_objectives = tuple(callable_objectives)

        self._initial_x = tuple(info.value for info in simulation.parameter_data)
        parameters = tuple(
            replace(info, value=value, lower_bound=lower, upper_bound=upper)
            for info, value, lower, upper in zip(
                simulation.parameter_data, x, problem.x_lower, problem.x_upper, strict=True)
        )
        self._simulation = Simulation(
            element_data=simulation.element_data,
            parameter_data=parameters,
            callable_constraints=callable_constraints,
            constraint_descriptors=constraint_descriptors,
            constraint_program=constraint_program,
            optics_program=simulation._optics_program,
            optical_stages=simulation._optical_stages,
            optical_inputs=simulation._optical_inputs,
            bindings=simulation.bindings,
        )

    @property
    def simulation(self):
        """Separate simulation with solved initial values and current-solve bounds."""
        return self._simulation

    @property
    def x(self):
        return self._x

    @property
    def cost(self):
        return self._result.cost

    @property
    def success(self):
        return self._result.success

    @property
    def message(self):
        return self._result.message

    @property
    def iterations(self):
        """Optimizer iterations, or None if the backend does not report them."""
        return self._result.iterations

    @property
    def function_evaluations(self):
        """Objective evaluations, or None if the backend does not report them."""
        return self._result.function_evaluations

    @property
    def feasibility_tolerance(self):
        return self._feasibility_tolerance

    @cached_property
    def _optical_result(self):
        return self.simulation.run(self.x)

    def run(self):
        """Return the optical result at the solved parameter vector."""
        return self._optical_result

    def value(self, quantity):
        """Return a design Parameter, or an expression of them, at the solution.

        A Parameter that was variable in the model (or a handle directly holding
        one) returns its solved value. Any other graph expression is evaluated
        with the interpreter: solved values replace the model's variable
        Parameters, fixed Parameters keep their current value, and the
        Simulation's bindings resolve its Symbols. KeyError is raised for
        anything the solution cannot determine, namely a variable Parameter
        outside the model, an unbound Symbol (its default is ignored), or a
        runtime result quantity (read those from run()).
        """
        if isinstance(quantity, Parameter) or (
            isinstance(quantity, InputNode) and isinstance(quantity.node, Parameter)
        ):
            return self.simulation.initial_values[quantity]
        if not isinstance(quantity, Node):
            raise TypeError(
                f"Expected a Parameter or graph expression, got {type(quantity).__name__}."
            )

        indices = self.simulation._optics_program.parameter_indices
        bindings = self.simulation.bindings
        parameter_values = {pid: self.x[index] for pid, index in indices.items()}
        for node in walk_ast((quantity,), bindings):
            if isinstance(node, Parameter) and id(node) not in indices:
                if node.is_variable:
                    raise KeyError(
                        f"{node.full_name} is a variable Parameter outside this solution's model, "
                        "so it has no solved value."
                    )
                parameter_values[id(node)] = node.value
            if isinstance(node, Symbol) and node.key not in bindings:
                if isinstance(node.key, ResultSymbolKey):
                    raise KeyError(
                        f"Symbol {node.key!r} is a runtime result quantity; read it from run()."
                    )
                raise KeyError(f"Symbol {node.key!r} has no binding in this solution's simulation.")

        value = np.array(evaluate_ast(quantity, bindings, parameter_values))
        return value.item() if value.ndim == 0 else value

    def _evaluate(self, program, descriptors, callables, description):
        bindings = {key: resolve_symbol(key, self.run()) for key in program.runtime_symbols}
        values = program.evaluate(self.x, bindings)
        entries = [(descriptor.position, descriptor, value)
                   for descriptor, value in zip(descriptors, values, strict=True)]
        if callables:
            view = self.run().view()
            for position, entry in callables:
                value = entry.evaluate(view)
                if isinstance(value, Node):
                    raise TypeError("Deferred solver callables must return numerical scalars or arrays.")
                entries.append((position, entry, value))
        entries.sort(key=lambda entry: entry[0])
        evaluated = []
        for position, entry, value in entries:
            value = np.array(value, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{description} {entry.label!r} evaluated to a non-finite value.")
            value.setflags(write=False)
            evaluated.append((position, entry, value))
        return evaluated

    @cached_property
    def objectives(self):
        results = []
        for position, entry, value in self._evaluate(
            self._objective_program, self._objective_descriptors,
            self._callable_objectives, "Objective",
        ):
            cost = entry.weight * np.sum(np.square(value))
            if not np.isfinite(cost):
                raise ValueError(f"Objective {entry.label!r} cost is non-finite.")
            results.append(ObjectiveResult(position, entry.label, value, entry.weight, float(cost)))
        return tuple(results)

    @cached_property
    def constraints(self):
        results = []
        for position, entry, value in self._evaluate(
            self.simulation._constraint_program, self.simulation._constraint_descriptors,
            self.simulation._callable_constraints, "Constraint",
        ):
            margin = np.minimum(value - entry.lower_bound, entry.upper_bound - value)
            violation = np.maximum(-margin, 0.0)
            results.append(ConstraintResult(
                position=position,
                label=entry.label,
                description=getattr(entry, "description", str(entry)),
                value=value,
                lower_bound=entry.lower_bound,
                upper_bound=entry.upper_bound,
                violation=violation,
                margin=margin,
                satisfied=bool(np.all(violation <= self.feasibility_tolerance)),
            ))
        return tuple(results)

    @cached_property
    def violations(self):
        """Violated model constraints, followed by violated parameter bounds."""
        violations = [entry for entry in self.constraints if not entry.satisfied]
        for info, value in zip(self.simulation.parameter_data, self.x, strict=True):
            margin = min(value - info.lower_bound, info.upper_bound - value)
            violation = max(-margin, 0.0)
            if violation > self.feasibility_tolerance:
                violations.append(ParameterBoundResult(
                    parameter=info, value=value, lower_bound=info.lower_bound,
                    upper_bound=info.upper_bound, violation=violation,
                    margin=margin, satisfied=False,
                ))
        return tuple(violations)

    @property
    def feasible(self):
        return not self.violations

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

        violations = self.violations
        status = "converged" if self.success else "not converged"
        feasibility = "feasible" if self.feasible else f"infeasible ({len(violations)} violated)"
        counts = ", ".join(
            f"{name}: {value}" for name, value in
            (("iterations", self.iterations), ("function evaluations", self.function_evaluations))
            if value is not None
        )
        lines = [
            f"Solution: {status}, {feasibility}",
            f"Cost: {self.cost:.4g}" + (f"    {counts}" if counts else ""),
            f"Message: {self.message}",
        ]

        if self.simulation.parameter_data:
            rows = [
                (info.name or "-", f"{initial:.4g}", f"{value:.4g}",
                 f"[{info.lower_bound:.4g}, {info.upper_bound:.4g}]")
                for info, initial, value in zip(
                    self.simulation.parameter_data, self._initial_x, self.x, strict=True)
            ]
            lines.extend(["", "Variables", "", *render_table(("Parameter", "Initial", "Solved", "Bounds"), rows)])

        if self.objectives:
            rows = [(entry.label or f"Objective {entry.position}", f"{entry.cost:.4g}") for entry in self.objectives]
            lines.extend(["", "Objectives", "", *render_table(("Objective", "Cost"), rows)])

        if self.constraints:
            rows = [
                (entry.label or entry.description, f"{float(np.min(entry.margin)):.4g}",
                 "ok" if entry.satisfied else "VIOLATED")
                for entry in self.constraints
            ]
            lines.extend(["", "Constraints", "", *render_table(("Constraint", "Margin", "Status"), rows)])

        return "\n".join(lines)
