from __future__ import annotations

from ..simulation import Simulation
from .solution import Solution
from .objective import Objective
from .constraint import Constraint, normalize_to_constraint



class SolverContext:
    def __init__(self, theta, simulation):
        self.theta = theta
        self.simulation = simulation
        self._result = None

    @property
    def result(self):
        if self._result is None:
            self._result = self.simulation.run(self.theta)
        return self._result

    def __getattr__(self, name):
        try:
            return self.simulation.context[name]
        except KeyError:
            raise AttributeError(name) from None


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
        self._node_cache = {}

    def target(self, *objectives):
        """Add soft optimization objectives."""
        self.objectives.extend(map(normalize_to_objective, objectives))
        return self

    def require(self, *constraints):
        """Add hard optimization constraints."""
        self.constraints.extend(map(normalize_to_constraint, constraints))
        return self

    def _evaluate_node(self, node, theta):
        key = id(node)

        if key not in self._node_cache:
            self._node_cache[key] = (node, self._compile_node(node))

        return self._node_cache[key][1](theta)

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
        def evaluate(context):
            result = func(context)
            if isinstance(result, Node):
                return self._evaluate_node(result, context.theta)
            return result
        return evaluate


    def _compile_objectives(self):
        objectives = []

        for obj in self.objectives:
            func = obj.evaluate

            if isinstance(func, Node):
                node_func = self._compile_node(func)
                evaluate = lambda ctx, fn=node_func: fn(ctx.theta)
            else:
                evaluate = self._compile_callable(func)

            objectives.append(
                Objective(evaluate=evaluate, weight=obj.weight, label=obj.label)
            )

        return tuple(objectives)


    def _compile_constraints(self):
        constraints = []

        for c in self.constraints:
            func = c.evaluate

            if isinstance(func, Node):
                node_func = self._compile_node(func)
                evaluate = lambda ctx, fn=node_func: fn(ctx.theta)
            else:
                evaluate = self._compile_callable(func)

            constraints.append(
                Constraint(
                    evaluate=evaluate,
                    lower_bound=c.lower_bound,
                    upper_bound=c.upper_bound,
                    label=c.label,
                )
            )

        return tuple(constraints)

    def _construct_context(self, theta):
        return SolveContext(theta, self.simulation)

    def solve(self) -> Solution:
        """Solve the inverse-design problem."""
        assert all(isinstance(x, Objective) for x in self.objectives)
        assert all(isinstance(x, Constraint) for x in self.constraints)

        constraints = self._compile_constraints()
        objectives = self._compile_objectives()

        def objective_function(theta):
            context = self._construct_context(theta)

            return sum(
                obj.weight * np.sum(np.square(obj.evaluate(context)))
                for obj in objectives
            )

        def constraint_function(theta):
            context = self._construct_context(theta)

            values = [
                np.atleast_1d(constraint.evaluate(context))
                for constraint in constraints
            ]
            return np.concatenate(values)

        raise NotImplementedError