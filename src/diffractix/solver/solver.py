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
        self.objectives.extend(normalize_to_objective(objective) for objective in objectives)
        return self

    def require(self, *constraints):
        """Add hard optimization constraints."""
        self.constraints.extend(normalize_to_constraint(constraint) for constraint in constraints)
        return self

    def solve(self) -> Solution:
        """Solve the inverse-design problem."""
        raise NotImplementedError