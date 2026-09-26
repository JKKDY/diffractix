from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import autograd.numpy as np

from .result import OptimizationResult
from .context import SolverContext
from .problem import Problem
from .objective import Objective
from .constraint import Constraint
from ..system.system import ParameterInfo

from diffractix.graph import evaluate_ast


DEFAULT_FEASIBILITY_TOLERANCE = 1e-8

@dataclass(frozen=True)
class SolvedParameter:
    info: ParameterInfo
    value: float

    @property
    def initial(self):
        return self.info.value

    @property
    def delta(self):
        return self.value - self.initial


@dataclass(frozen=True)
class ConstraintResult:
    constraint: Constraint
    value: object
    lower_bound: float
    upper_bound: float
    violation: object
    margin: object
    satisfied: bool


@dataclass(frozen=True)
class TargetResult:
    objective: Objective
    value: object
    weight: float
    cost: float


class Solution:
    """Result of solving a Diffractix inverse-design problem."""

    def __init__(
        self,
        result: OptimizationResult,
        problem: Problem,
        simulation,
        objectives,
        constraints,
        feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE
    ):
        if not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 0:
            raise ValueError(
                "feasibility_tolerance must be finite and non-negative."
            )

        if len(result.x) != len(simulation.initial_values):
            raise ValueError(
                "Backend result x length must match simulation.initial_values."
            )

        self._result = result
        self._problem = problem
        self._simulation = simulation
        self._objectives = tuple(objectives)
        self._constraints = tuple(constraints)
        self._feasibility_tolerance = feasibility_tolerance

    @cached_property
    def _context(self):
        return SolverContext(
            self.x,
            self._simulation.run,
        )


    @property
    def x(self):
        return self._result.x

    @cached_property
    def parameter_info(self):
        return tuple(sorted(
            (
                info for info in self._simulation.parameter_info.values()
                if info.parameter_index is not None
            ),
            key=lambda info: info.parameter_index,
        ))

    @cached_property
    def parameters(self):
        return tuple(
            SolvedParameter(info, value)
            for info, value in zip(self.parameter_info, self.x)
        )

    @property
    def success(self):
        return self._result.success

    @property
    def cost(self):
        return self._result.cost

    @property
    def message(self):
        return self._result.message

    @cached_property
    def constraints(self):
        results = []

        for c in self._constraints:
            val = np.asarray(c.evaluate(self._context))
            margin = np.minimum(val - c.lower_bound, c.upper_bound - val)
            violation = np.maximum(-margin, 0.0)

            results.append(
                ConstraintResult(
                    constraint=c,
                    value=val,
                    lower_bound=c.lower_bound,
                    upper_bound=c.upper_bound,
                    violation=violation,
                    margin=margin,
                    satisfied=bool(np.all(violation <= self.feasibility_tolerance)),
                )
            )

        return tuple(results)

    @cached_property
    def targets(self):
        return tuple(
            TargetResult(
                objective=obj,
                value=(val := obj.evaluate(self._context)),
                weight=obj.weight,
                cost=obj.weight * np.sum(np.square(val)),
            )
            for obj in self._objectives
        )

    @cached_property
    def violations(self):
        return tuple(
            result for result in self.constraints
            if not result.satisfied
        )

    @property
    def feasibility_tolerance(self):
        return self._feasibility_tolerance

    @cached_property
    def feasible(self):
        return not self.violations

    def value_of(self, node):
        return evaluate_ast(
            node,
            self.x,
            parameter_snapshot=self._simulation.parameter_info,
        )

    def __getitem__(self, node):
        return self.value_of(node)

    def run(self):
        return self._context._get_result()


