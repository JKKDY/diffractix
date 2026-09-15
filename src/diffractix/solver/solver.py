from __future__ import annotations

import autograd.numpy as np

from diffractix.graph import (
    Node,
    Symbol,
    SymbolicControlFlowError,
    compile_ast,
    evaluate_ast,
)

from ..simulation import Simulation
from ..system import System
from .solution import Solution
from .objective import Objective, normalize_to_objective
from .constraint import Constraint, normalize_to_constraint
from .context import (
    SolverCompileContext,
    SolverContext,
    SolverSymbolKey,
    SolverSymbolKind,
)


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

        for constraint in self.constraints:
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


    def solve(self) -> Solution:
        """Solve the inverse-design problem."""

        assert all(isinstance(x, Objective) for x in self.objectives)
        assert all(isinstance(x, Constraint) for x in self.constraints)

        constraints = self._compile_constraints()
        objectives = self._compile_objectives()

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

        raise NotImplementedError
