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
    """One objective's value, weight and cost, at the solution.

    Attributes
    ----------
    position : int
        Index of the objective in the order it was given to :meth:`Solver.target`.
    label : str or None
        Name given to the objective, or ``None``.
    value : array_like
        The residual's value at the solution.
    weight : float
        Weight used in the total cost.
    cost : float
        ``weight * sum(value**2)``.
    """
    position: int
    label: str | None
    value: object
    weight: float
    cost: float


@dataclass(frozen=True)
class ConstraintResult:
    """One constraint's value and feasibility, at the solution.

    Attributes
    ----------
    position : int
        Index of the constraint among all the solved problem's constraints: the
        System's first, then the Solver's, each in the order they were given.
    label : str or None
        Name given to the constraint, or ``None``.
    description : str
        Readable description, used when `label` is ``None``.
    value : array_like
        The constrained quantity's value at the solution.
    lower_bound : float
        Lower bound.
    upper_bound : float
        Upper bound.
    violation : array_like
        How far `value` lies outside the bounds; 0 where it is within them.
    margin : array_like
        How far `value` lies inside the bounds; negative where it violates them.
    satisfied : bool
        Whether `violation` is within the solution's `feasibility_tolerance`.
    """
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
    """One variable Parameter's bound violation, at the solution.

    Attributes
    ----------
    parameter : ParameterData
        The Parameter's metadata, including its name.
    value : float
        Solved value.
    lower_bound : float
        Lower bound.
    upper_bound : float
        Upper bound.
    violation : float
        How far `value` lies outside the bounds; 0 where it is within them.
    margin : float
        How far `value` lies inside the bounds; negative where it violates them.
    satisfied : bool
        Whether the bound holds. Always ``False`` here, since a
        ParameterBoundResult is only created for a violated bound; the field
        matches `ConstraintResult`, so every entry of `Solution.violations` has it.
    """
    parameter: ParameterData
    value: float
    lower_bound: float
    upper_bound: float
    violation: float
    margin: float
    satisfied: bool


class Solution:
    """Solved Parameter values, and the model's objectives, constraints and bounds evaluated there.

    Returned by :meth:`Solver.solve`. All quantities are in physical SI
    coordinates, regardless of the optimizer's internal scaling.

    Parameters
    ----------
    result : SolverResult
        The backend's raw result.
    problem : Problem
        The scaled optimization problem that was solved.
    simulation : Simulation
        The Simulation the Solver optimized.
    objective_program, constraint_program : Program
        Compiled graph-expression objectives and constraints.
    objective_descriptors, constraint_descriptors : tuple
        Metadata for the compiled objectives and constraints.
    callable_objectives, callable_constraints : tuple
        Deferred objective and constraint callables, with their declaration
        positions.
    feasibility_tolerance : float, optional
        Maximum constraint or bound violation still counted as feasible. Defaults
        to `DEFAULT_FEASIBILITY_TOLERANCE`.

    Raises
    ------
    TypeError
        If `feasibility_tolerance` is not a real numeric scalar.
    ValueError
        If `feasibility_tolerance` is not finite or is negative, or the
        backend result does not match `simulation`.

    Examples
    --------
    >>> from diffractix import Parameter, Ray, Solver, Space, System
    >>> from diffractix.units import mm
    >>> distance = Parameter(0.05).variable()
    >>> solver = Solver(System(Space(d=distance), optical_input=Ray(x=0, theta=0.01, n=1)))
    >>> solver = solver.target(value=lambda sim: sim.final.x, target=2 * mm)
    >>> solution = solver.solve()
    >>> solution.success, round(float(solution.value(distance)), 6)
    (True, 0.2)
    """

    def __init__(
        self,
        result: SolverResult,
        problem,
        simulation: Simulation,
        *,
        objective_program,
        objective_descriptors,
        callable_objectives,
        constraint_program,
        constraint_descriptors,
        callable_constraints,
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
        """Simulation: a copy of the optimized Simulation, with the solved values as initial values and the current bounds."""
        return self._simulation

    @property
    def x(self):
        """x: the solved values of the variable Parameters (theta), in
        `simulation.parameter_data` order."""
        return self._x

    @property
    def cost(self):
        """Cost: the total objective cost at the solution."""
        return self._result.cost

    @property
    def success(self):
        """Success: whether the optimizer reports successful termination."""
        return self._result.success

    @property
    def message(self):
        """Message: the termination message from the optimizer."""
        return self._result.message

    @property
    def iterations(self):
        """Iterations: optimizer iterations, or ``None`` if the backend does not report them."""
        return self._result.iterations

    @property
    def function_evaluations(self):
        """Function evaluations: objective evaluations, or ``None`` if the backend does not report them."""
        return self._result.function_evaluations

    @property
    def feasibility_tolerance(self):
        """Feasibility tolerance: the maximum constraint or bound violation still counted as feasible."""
        return self._feasibility_tolerance

    @cached_property
    def _optical_result(self):
        return self.simulation.run(self.x)

    def run(self):
        """Evaluate the simulation at the solved parameter vector.

        Returns
        -------
        SimulationResult
            The result; cached, so repeated calls return the same object.
        """
        return self._optical_result

    def value(self, quantity):
        """Return a design Parameter, or an expression of them, evaluated at the solution.

        Parameters
        ----------
        quantity : Parameter, InputNode or Node
            A variable Parameter in the model (or its direct handle) to read its
            solved value, or any other graph expression to evaluate, such as a
            derived gap.

        Returns
        -------
        float or array_like
            The value. Solved values replace the model's variable Parameters, fixed
            Parameters keep their current value, and the Simulation's bindings
            resolve its Symbols.

        Raises
        ------
        TypeError
            If `quantity` is not a Parameter or graph expression.
        KeyError
            If `quantity` depends on a variable Parameter outside the model, an
            unbound Symbol, or a runtime result quantity;
            read that from :meth:`run`.
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
                value = entry.quantity(view)
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
        """Objectives: one `ObjectiveResult` per objective, evaluated at the solution."""
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
        """Constraints: one `ConstraintResult` per constraint, evaluated at the solution.

        Includes the Simulation's own generated constraints, such as refractive
        index continuity.
        """
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
        """Violations: constraints and Parameter bounds the solution does not satisfy within `feasibility_tolerance`.

        Violated constraints come first, as `ConstraintResult`, followed by violated
        Parameter bounds as `ParameterBoundResult`.
        """
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
        """Feasible: whether the solution has no `violations`."""
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
