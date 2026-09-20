from __future__ import annotations

from enum import Enum, auto

import autograd.numpy as np
from autograd import grad, hessian, jacobian

from diffractix.graph import (
    Node,
    Symbol,
    SymbolicControlFlowError,
    compile_ast,
    evaluate_ast,
)

from ..simulation import Simulation
from ..system import System
from .solution import DEFAULT_FEASIBILITY_TOLERANCE, Solution
from .objective import Objective, normalize_to_objective
from .constraint import Constraint, normalize_to_constraint
from .problem import Problem
from .backends import solve_ipopt, solve_nlopt, solve_scipy
from .context import (
    SolverCompileContext,
    SolverContext,
    SolverSymbolKey,
    SolverSymbolKind,
)


class Backend(Enum):
    """Optimization backends supported by :class:`Solver`."""

    IPOPT = auto()
    SCIPY = auto()
    NLOPT = auto()


class Solver:
    """Solve inverse desing problem"""

    def __init__(self, simulation: Simulation | System):
        if isinstance(simulation, System):
            simulation = simulation.build()

        if not isinstance(simulation, Simulation):
            raise TypeError("simulation must be a Simulation or System.")

        self.simulation = simulation
        self.objectives = []
        self.constraints = []

    def target(self, *objectives):
        """Add soft optimization objectives."""
        self.objectives.extend(map(normalize_to_objective, objectives))
        return self

    def require(self, *constraints):
        """Add hard optimization constraints."""
        self.constraints.extend(map(normalize_to_constraint, constraints))
        return self


    def _resolve_symbol(
        self,
        key: SolverSymbolKey,
        compile_context: SolverCompileContext,
        context: SolverContext,
    ):
        if key.kind is SolverSymbolKind.AT:
            element = compile_context._target_elements[key.element_id]
            return getattr(context.at(element, key.occurrence), key.name)

        if key.kind is SolverSymbolKind.AFTER:
            element = compile_context._target_elements[key.element_id]
            return getattr(context.after(element, key.occurrence), key.name)

        if key.kind is SolverSymbolKind.Z_AT:
            element = compile_context._target_elements[key.element_id]
            return context.z_at(element, key.occurrence)

        if key.kind is SolverSymbolKind.Z_AFTER:
            element = compile_context._target_elements[key.element_id]
            return context.z_after(element, key.occurrence)

        if key.kind is SolverSymbolKind.Z:
            return context.z[key.index]

        if key.kind is SolverSymbolKind.STATE:
            return getattr(context.states[key.index], key.name)

        raise KeyError(f"Unknown solver Symbol key {key!r}.")


    def _compile_node(self, node, compile_context=None):
        graph = compile_ast(
            (node,),
            context=self.simulation.simulation_context,
            parameter_snapshot=self.simulation.parameter_info,
        )

        indices = []
        for parameter in graph.variables:
            info = self.simulation.parameter_info.get(id(parameter))
            if info is None or info.parameter_index is None:
                raise ValueError(
                    f"Variable parameter {parameter.name!r} "
                    "is not part of this Simulation."
                )
            indices.append(info.parameter_index)

        indices = tuple(indices)

        if graph.symbols and compile_context is None:
            raise ValueError(
                "Expression contains runtime Symbols but no compile context is available to resolve them."
            )

        def evaluate(context):
            local_theta = (
                np.stack([context.theta[i] for i in indices])
                if indices else np.array([])
            )

            bindings = {
                symbol.key: self._resolve_symbol(symbol.key, compile_context, context)
                for symbol in graph.symbols
            }

            return graph.evaluate(local_theta, bindings=bindings)[0]

        return evaluate


    def _compile_callable(self, func):
        compile_context = SolverCompileContext()

        try:
            result = func(compile_context)

            if isinstance(result, Node):
                return self._compile_node(result, compile_context=compile_context)

            return lambda context, value=result: value

        except SymbolicControlFlowError:
            def evaluate(context):
                result = func(context)

                if isinstance(result, Node):
                    return evaluate_ast(
                        result,
                        context.theta,
                        parameter_snapshot=self.simulation.parameter_info,
                    )
                return result

            return evaluate


    def _compile_objectives(self):
        compiled_objectives = []

        for objective in self.objectives:
            func = objective.evaluate
            evaluate = self._compile_node(func) if isinstance(func, Node) else self._compile_callable(func)

            compiled_objectives.append(
                Objective(evaluate=evaluate, weight=objective.weight, label=objective.label,)
            )

        return tuple(compiled_objectives)


    def _compile_constraints(self):
        compiled_constraints = []

        requirements = tuple(
            normalize_to_constraint(requirement)
            for requirement in getattr(self.simulation, "requirements", ())
        )

        for constraint in (*requirements, *self.constraints):
            func = constraint.evaluate
            evaluate = self._compile_node(func) if isinstance(func, Node) else self._compile_callable(func)

            compiled_constraints.append(
                Constraint(
                    evaluate=evaluate,
                    lower_bound=constraint.lower_bound,
                    upper_bound=constraint.upper_bound,
                    label=constraint.label,
                )
            )

        return tuple(compiled_constraints)

    def _constraint_bounds(self, constraints):
        context = SolverContext(self.simulation.initial_values, self.simulation.run)

        lower, upper = [], []
        for c in constraints:
            size = np.size(c.evaluate(context))
            lower.extend([c.lower_bound] * size)
            upper.extend([c.upper_bound] * size)

        return np.asarray(lower), np.asarray(upper)

    def _parameter_bounds(self):
        variables = [
            info for info in self.simulation.parameter_info.values()
            if info.parameter_index is not None
        ]
        variables.sort(key=lambda info: info.parameter_index)

        lower_bounds = np.array([info.lower_bound for info in variables])
        upper_bounds = np.array([info.upper_bound for info in variables])

        return lower_bounds, upper_bounds

    def solve(
        self,
        backend: Backend = Backend.SCIPY,
        method=None,
        options: dict | None = None,
        feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE
    ) -> Solution:
        """Solve the inverse-design problem."""

        assert all(isinstance(x, Objective) for x in self.objectives)
        assert all(isinstance(x, Constraint) for x in self.constraints)

        options = {} if options is None else dict(options)

        constraints = self._compile_constraints()
        objectives = self._compile_objectives()
        parameter_lower, parameter_upper = self._parameter_bounds()
        constraint_lower, constraint_upper = self._constraint_bounds(constraints)

        def objective_function(theta):
            context = SolverContext(theta, self.simulation.run)
            return sum(
                obj.weight * np.sum(np.square(obj.evaluate(context)))
                for obj in objectives
            )

        def constraint_function(theta):
            context = SolverContext(theta, self.simulation.run)
            values = [
                np.atleast_1d(constraint.evaluate(context))
                for constraint in constraints
            ]
            return np.concatenate(values) if values else np.array([])

        def weighted_constraints(theta, multipliers):
            return np.dot(multipliers, constraint_function(theta))

        problem = Problem(
            x0=self.simulation.initial_values,
            x_lower=parameter_lower,
            x_upper=parameter_upper,
            objective=objective_function,
            gradient=grad(objective_function),
            objective_hessian=hessian(objective_function),
            constraints=constraint_function,
            jacobian=jacobian(constraint_function),
            constraint_hessian=hessian(weighted_constraints, 0),
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
        )

        match backend:
            case Backend.IPOPT: solve = solve_ipopt
            case Backend.SCIPY: solve = solve_scipy
            case Backend.NLOPT: solve = solve_nlopt
            case _: raise ValueError(f"Unsupported backend: {backend}")

        result = solve(problem, method=method, options=options)

        return Solution(
            result,
            problem,
            self.simulation,
            objectives,
            constraints,
            feasibility_tolerance,
        )
