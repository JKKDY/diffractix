from __future__ import annotations

from enum import Enum
from math import isfinite
from numbers import Real
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from diffractix.backend import xp, grad, hessian, jacobian, jit, require

from diffractix.graph import (
    Node,
    Program,
    remap_parameters,
    remap_runtime_symbols,
    fuse,
    optimize,
)

from diffractix.simulation import Simulation
from diffractix.simulation.result import SimulationResultTracer, resolve_symbol
from diffractix.system import System
from .solution import DEFAULT_FEASIBILITY_TOLERANCE, Solution
from .objective import Objective, normalize_to_objective, prepare_objectives
from .constraint import (
    Constraint,
    normalize_to_constraint,
    prepare_constraints,
    reduce_constraints,
)
from .problem import Problem
from .backends import solve_ipopt, solve_nlopt, solve_scipy


class Backend(str, Enum):
    """Optimization backends supported by :class:`Solver`."""

    IPOPT = "ipopt"
    SCIPY = "scipy"
    NLOPT = "nlopt"



class Solver:
    """Solve one fixed optimization model; create a new Solver to change it.

    Traceable callables may return Nodes. Deferred callables receive a
    numerical result view and must return numerical scalars or arrays.
    """

    @dataclass(frozen=True)
    class Callbacks:
        objective: Callable
        gradient: Callable
        objective_hessian: Callable

        constraints: Callable
        jacobian: Callable
        constraint_hessian: Callable


    @dataclass
    class SolverCache:
        callbacks:                 Callbacks

        objective_program:         Program
        objective_descriptors:     tuple[ObjectiveDescriptor, ...]
        callable_objectives:       tuple[tuple[int, Objective], ...]

        constraint_program:        Program
        constraint_descriptors:    tuple[ConstraintDescriptor, ...]
        callable_constraints:      tuple[tuple[int, Constraint], ...]

        graph_constraint_rows:     tuple[range, ...]
        callable_constraint_rows:  tuple[range, ...]

        parameter_lower:           np.ndarray | None = None
        parameter_upper:           np.ndarray | None = None

        output_indices:            tuple[int, ...] | None = None
        active_descriptors:        tuple[ConstraintDescriptor, ...] | None = None

        reduced_callbacks:         Callbacks | None = None
        constraint_lower:          np.ndarray | None = None
        constraint_upper:          np.ndarray | None = None


    #-----------
    # PUBLIC API
    #-----------

    def __init__(self, simulation: Simulation | System):
        if isinstance(simulation, System):
            simulation = simulation.build()

        if not isinstance(simulation, Simulation):
            raise TypeError("simulation must be a Simulation or System.")

        self._simulation = simulation
        self._cache = None
        self._cache_use_jit = None

        self._objectives = []
        self._constraints = []

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    def target(self, *objectives, value=None, target=None, scale=None, label=None):
        """Add residual objectives, or match a callable value to a target."""
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")

        if value is not None:
            if objectives:
                raise ValueError("value cannot be combined with positional objectives.")
            if not callable(value):
                raise TypeError("value must be a callable accepting the solver context.")
            if target is None:
                raise ValueError("target must be specified when value is supplied.")

            scale = 1 if scale is None else scale
            if scale == 0:
                raise ValueError("scale must be nonzero.")

            objectives = (Objective(
                lambda ctx: (value(ctx) - target) / scale,
                label=label,
            ),)
        elif target is not None or scale is not None:
            raise ValueError("target and scale require value.")
        elif label is not None:
            if len(objectives) != 1 or isinstance(objectives[0], Objective):
                raise ValueError("label requires a single residual Node or callable.")
            objectives = (Objective(objectives[0], label=label),)

        self._objectives.extend([normalize_to_objective(objective) for objective in objectives])
        return self

    def require(self, *constraints):
        """Add hard optimization constraints."""
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")

        normalized = tuple(normalize_to_constraint(item) for item in constraints)
        self._constraints.extend(normalized)
        return self

    def solve(
        self,
        backend: Backend = Backend.SCIPY,
        method=None,
        options: dict = {},
        feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
        use_jit = None,
        x0=None,
        x_lower=None,
        x_upper=None,
    ) -> Solution:
        """Solve the inverse-design problem."""
        n = len(self.simulation.optics_program.parameter_indices)
        
        if n == 0:
            raise ValueError("Optimization problem must have at least one variable parameter.")
        if isinstance(feasibility_tolerance, bool) or not isinstance(feasibility_tolerance, Real):
            raise TypeError("feasibility_tolerance must be a real numeric scalar.")
        if not isfinite(feasibility_tolerance) or feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and non-negative.")
        if not all(isinstance(item, Objective) for item in self._objectives):
            raise TypeError("Solver objectives must contain only Objective instances.")
        if not all(isinstance(item, Constraint) for item in self._constraints):
            raise TypeError("Solver constraints must contain only Constraint instances.")
        if not (
            self._objectives or self._constraints
            or self.simulation.constraint_descriptors or self.simulation.callable_constraints
        ):
            raise ValueError("Optimization problem must have at least one objective or constraint.")


        initial_values = np.array(
            self.simulation.initial_values.values if x0 is None else x0, dtype=float,
        )
        parameter_lower = np.array(
            x_lower if x_lower is not None 
                else [item.lower_bound for item in self.simulation.parameter_data],
            dtype=float,
        )
        parameter_upper = np.array(
            x_upper if x_upper is not None 
                else [item.upper_bound for item in self.simulation.parameter_data],
            dtype=float,
        )


        for name, values in (("x0", initial_values), ("x_lower", parameter_lower), ("x_upper", parameter_upper)):
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


        if self._cache is None or (use_jit is not None and self._cache_use_jit != use_jit):
            use_jit = use_jit if use_jit is not None else False 
            self._cache = self._build_cache(use_jit, initial_values)
            self._cache_use_jit = use_jit

        cache = self._cache

        if (
            not np.array_equal(cache.parameter_lower, parameter_lower)
            or not np.array_equal(cache.parameter_upper, parameter_upper)
        ):
            self._update_reduction(cache, parameter_lower, parameter_upper)

        callbacks = cache.reduced_callbacks

        problem = Problem(
            x0=initial_values,
            x_lower=parameter_lower,
            x_upper=parameter_upper,

            objective=callbacks.objective,
            gradient=callbacks.gradient,
            objective_hessian=callbacks.objective_hessian,
            constraints=callbacks.constraints,
            jacobian=callbacks.jacobian,
            constraint_hessian=callbacks.constraint_hessian,

            constraint_lower=cache.constraint_lower,
            constraint_upper=cache.constraint_upper
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
            result, problem, self.simulation,
            objective_program=cache.objective_program,
            objective_descriptors=cache.objective_descriptors,
            callable_objectives=cache.callable_objectives,
            constraint_program=cache.constraint_program,
            constraint_descriptors=cache.constraint_descriptors,
            callable_constraints=cache.callable_constraints,
            feasibility_tolerance=feasibility_tolerance,
        )

    #-----------------
    # INTERNAL METHODS
    #-----------------
    def _finalize_constraints(self):
        """Align and fuse prepared constraints with the Simulation's constraints."""
        tracer = SimulationResultTracer(self.simulation.element_data, self.simulation.optical_inputs[0].initial_state)

        (
            solver_constraint_program, 
            solver_constraint_descriptors, 
            callable_solver_constraints
        ) = prepare_constraints(self._constraints, tracer=tracer, bindings=self.simulation.bindings)

        system_constraint_program = self.simulation.constraint_program
        system_constraint_descriptors = self.simulation.constraint_descriptors
        callable_system_constraints = self.simulation.callable_constraints

        # make both programs use the same parameter and runtime Symbol indices
        parameter_indices = self.simulation.optics_program.parameter_indices
        system_constraint_program = remap_parameters(system_constraint_program, parameter_indices)
        solver_constraint_program = remap_parameters(solver_constraint_program, parameter_indices)

        runtime_symbols = dict(system_constraint_program.runtime_symbols)
        for key in solver_constraint_program.runtime_symbols:
            if key not in runtime_symbols:
                runtime_symbols[key] = len(runtime_symbols)
        system_constraint_program = remap_runtime_symbols(system_constraint_program, runtime_symbols)
        solver_constraint_program = remap_runtime_symbols(solver_constraint_program, runtime_symbols)

        # fuse the programs and combine their metadata in the same order
        positions = [descriptor.position for descriptor in system_constraint_descriptors]
        positions.extend(position for position, _ in callable_system_constraints)
        offset = max(positions, default=-1) + 1
        solver_constraint_descriptors = tuple(
            replace(descriptor, position=descriptor.position + offset)
            for descriptor in solver_constraint_descriptors
        )
        callable_solver_constraints = tuple(
            (position + offset, constraint)
            for position, constraint in callable_solver_constraints
        )

        constraint_program = fuse(system_constraint_program, solver_constraint_program)
        constraint_descriptors = (*system_constraint_descriptors, *solver_constraint_descriptors)
        callable_constraints = (*callable_system_constraints, *callable_solver_constraints)

        return constraint_program, constraint_descriptors, callable_constraints


    def _finalize_objectives(self):
        """Trace objectives and align their program with the Simulation's theta."""
        tracer = SimulationResultTracer(self.simulation.element_data, self.simulation.optical_inputs[0].initial_state)
        program, descriptors, callables = prepare_objectives(
            self._objectives, tracer=tracer, bindings=self.simulation.bindings,
        )
        program = remap_parameters(program, self.simulation.optics_program.parameter_indices)
        return program, descriptors, callables


    def _build_cache(self, use_jit, initial_values):
        constraint_program, constraint_descriptors, callable_constraints = self._finalize_constraints()
        constraint_program = optimize(constraint_program)

        objective_program, objective_descriptors, callable_objectives = self._finalize_objectives()
        objective_program = optimize(objective_program)

        callbacks, graph_constraint_rows, callable_constraint_rows = self._build_callbacks(
            objective_program=objective_program,
            objective_descriptors=objective_descriptors,
            callable_objectives=callable_objectives,
            constraint_program=constraint_program,
            constraint_descriptors=constraint_descriptors,
            callable_constraints=callable_constraints,
            initial_values=initial_values,
            use_jit=use_jit,
        )

        return self.SolverCache(
            callbacks=callbacks,
            objective_program=objective_program,
            objective_descriptors=objective_descriptors,
            callable_objectives=callable_objectives,
            constraint_program=constraint_program,
            constraint_descriptors=constraint_descriptors,
            callable_constraints=callable_constraints,
            graph_constraint_rows=graph_constraint_rows,
            callable_constraint_rows=callable_constraint_rows,
        )


    def _build_callbacks(
        self, *, objective_program, objective_descriptors, callable_objectives,
        constraint_program, constraint_descriptors, callable_constraints,
        initial_values, use_jit,
    ):
        def evaluate_callable(entry, view):
            value = entry.evaluate(view)
            if isinstance(value, Node):
                raise TypeError(
                    "Deferred solver callables must return numerical scalars or arrays. "
                    "Received a symbolic Node."
                )
            return value

        def check_finite_value(value, description):
            require(xp.all(xp.isfinite(value)), f"{description} evaluated to a non-finite value.")
            return value

        objective_metadata = (*objective_descriptors, *(entry for _, entry in callable_objectives))
        objective_descriptions = tuple(
            "Objective" if entry.label is None else f"Objective {entry.label!r}"
            for entry in objective_metadata
        )
        split_objectives = len(objective_descriptors)

        def objective_cost(value, descriptor, description):
            value = check_finite_value(value, description)
            return descriptor.weight * xp.sum(xp.square(value))

        def objective(theta):
            result = None
            if objective_program.runtime_symbols or callable_objectives:
                result = self.simulation.run(theta)
            bindings = {key: resolve_symbol(key, result) for key in objective_program.runtime_symbols}
            values = objective_program.evaluate(theta, bindings)
            view = result.view() if callable_objectives else None

            compiled = sum(
                (objective_cost(value, descriptor, description)
                 for value, descriptor, description in zip(
                     values, objective_descriptors, objective_descriptions[:split_objectives])),
                0.0 * xp.sum(theta),
            )
            dynamic = sum(
                objective_cost(evaluate_callable(entry, view), entry, description)
                for (_, entry), description in zip(
                    callable_objectives, objective_descriptions[split_objectives:])
            )
            return compiled + dynamic

        constraint_metadata = (*constraint_descriptors, *(entry for _, entry in callable_constraints))
        constraint_descriptions = tuple(
            "Constraint" if entry.label is None else f"Constraint {entry.label!r}"
            for entry in constraint_metadata
        )

        def constraint_values(theta):
            result = None
            if constraint_program.runtime_symbols or callable_constraints:
                result = self.simulation.run(theta)
            bindings = {key: resolve_symbol(key, result) for key in constraint_program.runtime_symbols}
            values = constraint_program.evaluate(theta, bindings)
            if callable_constraints:
                view = result.view()
                values += tuple(evaluate_callable(entry, view) for _, entry in callable_constraints)
            for value, description in zip(values, constraint_descriptions):
                check_finite_value(value, description)
            return values

        def constraints(theta):
            values = constraint_values(theta)
            for value, expected_shape, description in zip(values, constraint_shapes, constraint_descriptions):
                if np.shape(value) != expected_shape:
                    raise ValueError(
                        f"{description} output shape changed from {expected_shape} to {np.shape(value)}."
                    )
            return xp.concatenate(tuple(xp.ravel(v) for v in values)) if values else xp.zeros((0,))

        def weighted_constraints(theta, multipliers):
            return xp.dot(multipliers, constraints(theta))

        values = constraint_values(initial_values)
        constraint_shapes = tuple(np.shape(value) for value in values)
        sizes = [np.size(value) for value in values]
        offsets = np.cumsum([0, *sizes])
        rows = tuple(range(start, end) for start, end in zip(offsets[:-1], offsets[1:]))

        split = len(constraint_descriptors)
        graph_constraint_rows = rows[:split]
        callable_constraint_rows = rows[split:]

        n = len(initial_values)
        m = sum(sizes)

        objective_gradient = grad(objective)
        objective_hessian = hessian(objective)
        constraint_jacobian = jacobian(constraints) if m else lambda theta: xp.zeros((0, n))
        constraint_hessian = hessian(weighted_constraints, 0) if m else lambda theta, multipliers: xp.zeros((n, n))

        full_objective = jit(objective) if use_jit else objective
        full_constraints = jit(constraints) if use_jit else constraints

        def checked_objective(theta):
            return check_finite_value(full_objective(theta), "Objective cost")

        def checked_constraints(theta):
            return check_finite_value(full_constraints(theta), "Constraint vector")

        callbacks = self.Callbacks(
            objective          = checked_objective,
            gradient           = jit(objective_gradient)  if use_jit else objective_gradient,
            objective_hessian  = jit(objective_hessian)   if use_jit else objective_hessian,
            constraints        = checked_constraints,
            jacobian           = jit(constraint_jacobian) if use_jit else constraint_jacobian,
            constraint_hessian = jit(constraint_hessian)  if use_jit else constraint_hessian,
        )

        return callbacks, graph_constraint_rows, callable_constraint_rows


    def _update_reduction(self, cache, parameter_lower, parameter_upper):
        cache.output_indices, cache.active_descriptors = reduce_constraints(
            cache.constraint_program,
            cache.constraint_descriptors,
            parameter_lower=parameter_lower,
            parameter_upper=parameter_upper,
        )

        graph_constraints = zip(
            cache.active_descriptors,
            (cache.graph_constraint_rows[i] for i in cache.output_indices),
        )
        callable_constraints = (
            (constraint, rows)
            for (_, constraint), rows in zip(cache.callable_constraints, cache.callable_constraint_rows)
        )
        active_constraints = (*graph_constraints, *callable_constraints)

        active_rows = np.array([row for _, rows in active_constraints for row in rows], dtype=int)
        full_size = sum(len(rows) for rows in (*cache.graph_constraint_rows, *cache.callable_constraint_rows))

        multiplier_map = xp.asarray(np.eye(full_size)[active_rows])
        full = cache.callbacks

        cache.reduced_callbacks = self.Callbacks(
            objective=full.objective,
            gradient=full.gradient,
            objective_hessian=full.objective_hessian,
            constraints=lambda theta: full.constraints(theta)[active_rows],
            jacobian=lambda theta: full.jacobian(theta)[active_rows],
            constraint_hessian=lambda theta, multipliers: full.constraint_hessian(
                theta, xp.dot(multipliers, multiplier_map)
            ),
        )

        cache.constraint_lower = np.array([
            c.lower_bound for c, rows in active_constraints for _ in rows
        ], dtype=float)
        cache.constraint_upper = np.array([
            c.upper_bound for c, rows in active_constraints for _ in rows
        ], dtype=float)

        cache.parameter_lower = parameter_lower.copy()
        cache.parameter_upper = parameter_upper.copy()
