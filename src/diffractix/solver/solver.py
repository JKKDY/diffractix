from __future__ import annotations

from ..simulation import Simulation
from .solution import Solution
from .objective import Objective
from .constraint import Constraint, normalize_to_constraint


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


    def _compile_node(self, node):
        graph = compile_ast((node,), context=self.simulation.context)

        indices = []
        for parameter in graph.variables:
            info = self.simulation.parameter_info.get(id(parameter))
            if info is None or info.parameter_index is None:
                raise ValueError(f"Variable parameter {parameter.name!r} is not part of this Simulation.")
            indices.append(info.parameter_index)

        indices = tuple(indices)

        def evaluate(theta):
            local_theta = np.stack([theta[i] for i in indices]) if indices else np.array([])
            return graph.evaluate(local_theta)[0]

        return evaluate

    def _compile_callable(self, func):
        def evaluate(theta):
            context = self._construct_context(theta)
            result = func(context)
            return context.value(result) if isinstance(result, Node) else result
        return evaluate

    def _compile_objectives(self):
        objectives = []

        for objective in self.objectives:
            func = objective.evaluate
            evaluate = self._compile_node(func) if isinstance(func, Node) else self._compile_callable(func)
            objectives.append(
                Objective(evaluate=evaluate, weight=objective.weight, label=objective.label)
            )

        return tuple(objectives)

    def _compile_constraints(self):
        constraints = []

        for constraint in self.constraints:
            func = constraint.evaluate
            evaluate = self._compile_node(func) if isinstance(func, Node) else self._compile_callable(func)
            constraints.append(
                Constraint(
                    evaluate=evaluate,
                    lower_bound=constraint.lower_bound,
                    upper_bound=constraint.upper_bound,
                    label=constraint.label,
                )
            )

        return tuple(constraints)

    def solve(self) -> Solution:
        """Solve the inverse-design problem."""
        assert all(isinstance(x, Objective) for x in self.objectives)
        assert all(isinstance(x, Constraint) for x in self.constraints)

        constraints = self._compile_constraints()
        objectives = self._compile_objectives()

        def objective_function(theta):
            return sum(
                objective.weight * np.sum(np.square(objective.evaluate(theta)))
                for objective in objectives
            )

        raise NotImplementedError