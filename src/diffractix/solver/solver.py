from __future__ import annotations

from ..simulation import Simulation
from .solution import Solution


class Solver:
    """Solve inverse desing problem"""

    def __init__(self, simulation: Simulation | System):
        self.simulation = simulation

    def target(self, *objectives):
        """Add soft optimization objectives."""
        raise NotImplementedError

    def require(self, *constraints):
        """Add hard optimization constraints."""
        raise NotImplementedError

    def solve(self) -> Solution:
        """Solve the inverse-design problem."""
        raise NotImplementedError