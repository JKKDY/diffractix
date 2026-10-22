from __future__ import annotations

from enum import Enum, auto
from math import isfinite
from numbers import Real
from dataclasses import dataclass
from typing import Callable

import numpy as np
from diffractix.backend import xp, grad, hessian, jacobian, jit

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
from .constraint import (
    Constraint,
    normalize_to_constraint,
    remove_bound_redundant_constraints,
    remove_affine_redundant_constraints,
    remove_duplicate_constraints
)
from .problem import Problem
from .backends import solve_ipopt, solve_nlopt, solve_scipy
from .context import (
    SolverCompileContext,
    SolverContext,
    SolverSymbolKey,
    SolverSymbolKind,
)

_UNSET = object()

class Backend(str, Enum):
    """Optimization backends supported by :class:`Solver`."""

    IPOPT = "ipopt"
    SCIPY = "scipy"
    NLOPT = "nlopt"


@dataclass(frozen=True)
class _Callbacks:
    objective: Callable
    gradient: Callable
    objective_hessian: Callable
    constraints: Callable
    jacobian: Callable
    constraint_hessian: Callable


@dataclass
class _SolverCache:
    compiled_objectives: tuple[Objective, ...]
    compiled_constraints: tuple[Constraint, ...]

    constraint_map: dict[int, tuple[Callable, range]]
    constraint_size: int  # number of scalar constraints

    callbacks: _Callbacks
    jit_callbacks: _Callbacks | None = None


class Solver:
    """Solve one fixed optimization model; create a new Solver to change it.

    Objectives and requirements are frozen when callbacks are first prepared.
    Do not mutate their ASTs or Python callable captures afterward.
    """

    def __init__(self, simulation: Simulation | System):
        if isinstance(simulation, System):
            simulation = simulation.build()

        if not isinstance(simulation, Simulation):
            raise TypeError("simulation must be a Simulation or System.")

        self.simulation = simulation
        self.objectives = []
        self.constraints = []
        self._cache = None

    @property
    def simulation(self):
        return self._simulation

    @simulation.setter
    def simulation(self, simulation):
        if hasattr(self, "_simulation"):
            raise AttributeError("Solver.simulation cannot be reassigned.")
        self._simulation = simulation


    def target(self, *objectives, value=_UNSET, target=_UNSET, scale=_UNSET, label=None):
        """Add residual objectives, or match a callable value to a target."""
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")
        if value is not _UNSET:
            if objectives:
                raise ValueError("value cannot be combined with positional objectives.")
            if not callable(value):
                raise TypeError("value must be a callable accepting the solver context.")
            if target is _UNSET:
                raise ValueError("target must be explicitly specified when value is supplied.")
            scale = 1 if scale is _UNSET else scale
            if scale == 0:
                raise ValueError("scale must be nonzero.")
            objectives = (Objective(lambda ctx: (value(ctx) - target) / scale, label=label),)
        elif target is not _UNSET or scale is not _UNSET:
            raise ValueError("target and scale require value.")
        elif label is not None:
            if len(objectives) != 1 or isinstance(objectives[0], Objective):
                raise ValueError("label requires a single residual Node or callable, not Objective instances.")
            objectives = (Objective(objectives[0], label=label),)
        self.objectives.extend(map(normalize_to_objective, objectives))
        return self

    def require(self, *constraints):
        """Add hard optimization constraints."""
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")
        self.constraints.extend(map(normalize_to_constraint, constraints))
        return self


    @staticmethod
    def _validate_finite_value(value, description):
        """Reject non-finite eager values."""
        try:
            is_finite = bool(np.all(np.isfinite(value)))
        except TypeError:
            # Autodiff tracers cannot necessarily be inspected by isfinite or
            # converted to bool. Defer validation until eager evaluation.
            return value

        if not is_finite:
            raise ValueError(f"{description} evaluated to a non-finite value.")

        return value


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
            if not -len(context.z) <= key.index < len(context.z):
                raise IndexError(f"Longitudinal coordinate index {key.index} out of range.")
            return context.z[key.index]

        if key.kind is SolverSymbolKind.STATE:
            return getattr(context.states[key.index], key.name)

        raise KeyError(f"Unknown solver Symbol key {key!r}.")


    def _compile_node(self, node, compile_context=None):
        graph = compile_ast(
            (node,),
            context=self.simulation.compile_context,
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
            local_theta = tuple(context.theta[i] for i in indices)

            bindings = {
                symbol.key: self._resolve_symbol(symbol.key, compile_context, context)
                for symbol in graph.symbols
            }

            return graph._evaluate_sequence(local_theta, bindings=bindings)[0]

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
                        bindings=self.simulation.execution_context,
                    )
                return result

            return evaluate


    def _compile_objectives(self, objectives):
        compiled_objectives = []

        for objective in objectives:
            func = objective.evaluate
            evaluate = self._compile_node(func) if isinstance(func, Node) else self._compile_callable(func)

            compiled_objectives.append(
                Objective(evaluate=evaluate, weight=objective.weight, label=objective.label,)
            )

        return tuple(compiled_objectives)


    def _compile_constraints(self, constraints):
        compiled_constraints = []

        for constraint in constraints:
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


    def _reduce_constraints(self, constraints, parameter_lower, parameter_upper):
        constraints = remove_bound_redundant_constraints(
            constraints,
            parameter_info=self.simulation.parameter_info,
            parameter_lower=parameter_lower,
            parameter_upper=parameter_upper,
        )

        constraints = remove_affine_redundant_constraints(
            constraints,
            parameter_info=self.simulation.parameter_info,
            parameter_lower=parameter_lower,
            parameter_upper=parameter_upper,
            compile_context=self.simulation.compile_context,
        )

        constraints = remove_duplicate_constraints(constraints)

        reduced = []
        indices = []

        for constraint in constraints:
            evaluate, rows = self._cache.constraint_map[id(constraint.evaluate)]

            reduced.append(Constraint(
                evaluate=evaluate,
                lower_bound=constraint.lower_bound,
                upper_bound=constraint.upper_bound,
                label=constraint.label,
            ))
            indices.extend(rows)

        return tuple(reduced), np.asarray(indices, dtype=int)


    def _constraint_bounds(self, constraints, theta=None):
        theta = self.simulation.initial_values if theta is None else theta
        context = SolverContext(theta, self.simulation.run)

        lower, upper = [], []
        for c in constraints:
            value = self._validate_finite_value(
                c.evaluate(context),
                "Constraint" if c.label is None else f"Constraint {c.label!r}"
            )
            size = np.size(value)
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

    def _build_cache(self, raw_objectives, raw_constraints) -> _SolverCache:
        n = len(self.simulation.initial_values)
        compiled_objectives = self._compile_objectives(raw_objectives)
        compiled_constraints = self._compile_constraints(raw_constraints)
        initial_context = SolverContext(self.simulation.initial_values, self.simulation.run)

        # Map each original evaluator to its compiled callback and scalar rows.
        constraint_map = {}
        constraint_shapes = []
        offset = 0

        for original, compiled in zip(raw_constraints, compiled_constraints):
            shape = np.shape(compiled.evaluate(initial_context))
            size = int(np.prod(shape))

            constraint_map.setdefault(
                id(original.evaluate),
                (compiled.evaluate, range(offset, offset + size)),
            )

            constraint_shapes.append(shape)
            offset += size

        def objective_function(theta):
            context = SolverContext(theta, self.simulation.run)
            return sum(
                obj.weight * xp.sum(xp.square(self._validate_finite_value(
                    obj.evaluate(context),
                    "Objective" if obj.label is None else f"Objective {obj.label!r}",
                )))
                for obj in compiled_objectives
            )

        def constraint_function(theta):
            context = SolverContext(theta, self.simulation.run)
            values = []

            for constraint, expected_shape in zip(compiled_constraints, constraint_shapes):
                value = constraint.evaluate(context)
                if np.shape(value) != expected_shape:
                    raise ValueError(
                        "Constraint output shape changed from "
                        f"{expected_shape} to {np.shape(value)}."
                    )
                value = self._validate_finite_value(
                    value,
                    "Constraint" if constraint.label is None
                    else f"Constraint {constraint.label!r}"
                )
                values.append(xp.ravel(value))

            return xp.concatenate(values) if values else xp.array([])

        def weighted_constraints(theta, multipliers):
            return xp.dot(multipliers, constraint_function(theta))

        return _SolverCache(
            compiled_objectives=compiled_objectives,
            compiled_constraints=compiled_constraints,
            constraint_map=constraint_map,
            constraint_size=offset,

            callbacks=_Callbacks(
                objective=objective_function,
                gradient=grad(objective_function),
                objective_hessian=hessian(objective_function),
                constraints=constraint_function,
                jacobian=jacobian(constraint_function) if offset else lambda theta: xp.zeros((0, n)),
                constraint_hessian=hessian(weighted_constraints, 0) if offset else lambda theta, multipliers: xp.zeros((n, n)),
            ),
        )


    def solve(
        self,
        backend: Backend = Backend.SCIPY,
        method=None,
        options: dict | None = None,
        feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
        use_jit = False,
        x0=None,
        x_lower=None,
        x_upper=None,
    ) -> Solution:
        """Solve the inverse-design problem."""
        options = {} if options is None else dict(options)
        n = len(self.simulation.initial_values)

        raw_constraints = (*self.simulation.requirements, *self.constraints)
        raw_objectives = self.objectives

        if isinstance(feasibility_tolerance, bool) or not isinstance(feasibility_tolerance, Real):
            raise TypeError("feasibility_tolerance must be a real numeric scalar.")
        if not isfinite(feasibility_tolerance):
            raise ValueError("feasibility_tolerance must be finite and non-negative.")
        if feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and non-negative.")

        if not all(isinstance(x, Objective) for x in raw_objectives):
            raise TypeError("Solver objectives must contain only Objective instances.")
        if not all(isinstance(x, Constraint) for x in raw_constraints):
            raise TypeError("Solver constraints must contain only Constraint instances.")

        if not raw_objectives and not raw_constraints:
            raise ValueError("Optimization problem must have at least one objective or constraint.")
        if len(self.simulation.initial_values) == 0:
            raise ValueError("Optimization problem must have at least one variable parameter.")

        default_lower, default_upper = self._parameter_bounds()
        initial_values = np.array(self.simulation.initial_values if x0 is None else x0, dtype=float)
        parameter_lower = np.array(default_lower if x_lower is None else x_lower, dtype=float)
        parameter_upper = np.array(default_upper if x_upper is None else x_upper, dtype=float)
        for name, values in (
            ("x0", initial_values),
            ("x_lower", parameter_lower),
            ("x_upper", parameter_upper),
        ):
            if values.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {values.shape}.")
        if not np.all(np.isfinite(initial_values)):
            raise ValueError("x0 must contain only finite values.")
        if np.any(np.isnan(parameter_lower)) or np.any(np.isnan(parameter_upper)):
            raise ValueError("Parameter bounds must not contain NaNs.")
        if np.any(parameter_lower > parameter_upper):
            raise ValueError("x_lower must not exceed x_upper.")
        if np.any(initial_values < parameter_lower) or np.any(initial_values > parameter_upper):
            raise ValueError("x0 must lie within the parameter bounds.")

        if self._cache is None:
            self._cache = self._build_cache(raw_objectives, raw_constraints)

        constraints, indices = self._reduce_constraints(raw_constraints, parameter_lower, parameter_upper)
        constraint_lower, constraint_upper = self._constraint_bounds(constraints, initial_values)

        callbacks = self._cache.callbacks
        if use_jit:
            if self._cache.jit_callbacks is None:
                self._cache.jit_callbacks = _Callbacks(
                    objective=jit(callbacks.objective),
                    gradient=jit(callbacks.gradient),
                    objective_hessian=jit(callbacks.objective_hessian),
                    constraints=jit(callbacks.constraints),
                    jacobian=jit(callbacks.jacobian),
                    constraint_hessian=jit(callbacks.constraint_hessian),
                )
            callbacks = self._cache.jit_callbacks

        constraints_fn = callbacks.constraints
        constraint_jacobian = callbacks.jacobian
        constraint_hessian = callbacks.constraint_hessian

        if len(indices) == 0:
            constraints_fn = lambda theta: xp.zeros((0,))
            constraint_jacobian = lambda theta: xp.zeros((0, n))
            constraint_hessian = lambda theta, multipliers: xp.zeros((n, n))

        elif not np.array_equal(indices, np.arange(self._cache.constraint_size)):
            full_constraints = constraints_fn
            full_jacobian = constraint_jacobian
            full_hessian = constraint_hessian
            multiplier_map = np.eye(self._cache.constraint_size)[indices]

            def constraints_fn(theta):
                return full_constraints(theta)[indices]

            def constraint_jacobian(theta):
                return full_jacobian(theta)[indices]

            def constraint_hessian(theta, multipliers):
                full_multipliers = xp.dot(multipliers, multiplier_map)
                return full_hessian(theta, full_multipliers)

        problem = Problem(
            x0=initial_values,
            x_lower=parameter_lower,
            x_upper=parameter_upper,
            objective=callbacks.objective,
            gradient=callbacks.gradient,
            objective_hessian=callbacks.objective_hessian,
            constraints=constraints_fn,
            jacobian=constraint_jacobian,
            constraint_hessian=constraint_hessian,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
        )

        match backend:
            case Backend.IPOPT: solve = solve_ipopt
            case Backend.SCIPY: solve = solve_scipy
            case Backend.NLOPT: solve = solve_nlopt
            case _: raise ValueError(f"Unsupported backend: {backend}")

        scaled_problem = problem.scale()
        result = solve(scaled_problem, method=method, options=options)
        result = problem.scaling.unscale(result)

        return Solution(
            result,
            problem,
            self.simulation,
            self._cache.compiled_objectives,
            self._cache.compiled_constraints,
            feasibility_tolerance,
        )
