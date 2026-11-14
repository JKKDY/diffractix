from __future__ import annotations

from enum import Enum
from math import isfinite
from numbers import Real
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

import numpy as np
from contextlib import contextmanager

from diffractix.backend import xp, grad, hessian, jacobian, require, tracing_errors
from diffractix.backend import jit as backend_jit

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
    label_constraints,
    normalize_to_constraint,
    prepare_constraints,
    reduce_constraints,
)
from .problem import Problem

if TYPE_CHECKING:
    from .constraint import ConstraintDescriptor
    from .objective import ObjectiveDescriptor
from .optimizers import solve_ipopt, solve_nlopt, solve_scipy


@contextmanager
def _explain_tracing_errors(jit: bool):
    """Add the jit=False hint to errors from callables that cannot be traced."""
    try:
        yield
    except tracing_errors() as error:
        if not jit:
            raise
        raise TypeError(
            "A callable used a value that is unknown while JIT traces it (Python control "
            "flow such as `if`, or float()/int()/np.asarray on a traced value). Pass "
            "jit=False to solve() to evaluate it eagerly, or write the branch with "
            f"xp.where (from diffractix.backend import xp). Original error: {str(error).splitlines()[0]}"
        ) from error


class Optimizer(str, Enum):
    """Optimization libraries supported by `Solver`; strings such as ``"scipy"`` are accepted too."""

    IPOPT = "ipopt"
    SCIPY = "scipy"
    NLOPT = "nlopt"



class Solver:
    """An optimization model built from a Simulation, with objectives and constraints solved together for one theta.

    Add objectives with :meth:`target` and hard constraints with :meth:`require`,
    then call :meth:`solve`. The model is fixed at the first :meth:`solve`; add
    more objectives or constraints to a new Solver instead.

    Parameters
    ----------
    simulation : Simulation or System
        The optical model to optimize. A System is built first.

    Raises
    ------
    TypeError
        If `simulation` is not a Simulation or System.

    Notes
    -----
    A traceable objective or constraint (a graph expression, or a callable that
    returns one from `simulation.symbolic`) is compiled with the rest of the
    model. A deferred callable, one whose Python control flow depends on result
    values and so cannot be traced, receives a numerical `SimulationResultView`
    instead and must return numerical scalars or arrays; it still needs a
    gradient, so it must be differentiable by the numerical backend.

    Examples
    --------
    >>> from diffractix import Parameter, Ray, Solver, Space, System
    >>> from diffractix.units import mm
    >>> distance = Parameter(0.05).variable()
    >>> solver = Solver(System(Space(d=distance), optical_input=Ray(x=0, theta=0.01, n=1)))
    >>> solver = solver.target(value=lambda sim: sim.final.x, target=2 * mm)
    >>> solution = solver.solve()
    >>> round(float(solution.value(distance)), 6)
    0.2
    """

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
        callbacks:                 Solver._Callbacks

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

        reduced_callbacks:         Solver._Callbacks | None = None
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
        self._cache_jit = None

        self._objectives = []
        self._constraints = []

    @property
    def simulation(self) -> Simulation:
        """Simulation: the optical model this Solver optimizes."""
        return self._simulation

    def target(self, *objectives, value=None, target=None, scale=None, weight=None, label=None):
        """Add residual objectives, or match a value to a target.

        Pass `value` and `target` to match a single quantity, or pass `Objective`
        instances, graph expressions or zero-argument callables as `*objectives`.

        Parameters
        ----------
        *objectives : Node, Objective or callable
            Residuals to minimize, each contributing ``weight * sum(residual**2)`` to
            the total cost. A graph expression, a zero-argument callable returning
            one, or a callable accepting the result (traceable, or deferred if it
            branches on result values); a plain callable is wrapped in an
            `Objective` with the given `weight` and `label`. Combining these with
            `value` is not allowed.
        value : Node or callable, optional
            Quantity to match to `target`: a graph expression, or a callable
            accepting the result. Requires `target`.
        target : float or array_like, optional
            Target value for `value`. Required together with `value`.
        scale : float or array_like, optional
            Scale for the residual ``(value - target) / scale``. Defaults to
            ``abs(target)`` where nonzero and to 1 where `target` is zero; give it
            explicitly for a zero target. Must be nonzero.
        weight : float, optional
            Multiplies the squared residual in the total cost; applies to `value` or
            to a single objective in `*objectives`. Defaults to 1.
        label : str, optional
            Name for a single objective, used in reports; applies to `value` or to a
            single objective in `*objectives`.

        Returns
        -------
        Solver
            ``self``, so calls can be chained.

        Raises
        ------
        RuntimeError
            If the model is already fixed by a previous :meth:`solve`.
        TypeError
            If `value` is given but is not a graph expression or callable.
        ValueError
            If `value` is combined with `*objectives`, `target` or `scale` is given
            without `value`, `scale` is zero, or `label` or `weight` is given
            together with more than one objective in `*objectives`, or with a
            single objective that is already an `Objective` instance.

        Notes
        -----
        A deferred callable must be differentiable by the numerical backend;
        diffractix does not check this.
        """
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")

        if value is not None:
            if objectives:
                raise ValueError("value cannot be combined with positional objectives.")
            if not (isinstance(value, Node) or callable(value)):
                raise TypeError("value must be a graph expression or a callable accepting the result.")
            if target is None:
                raise ValueError("target must be specified when value is supplied.")

            if scale is None:
                magnitude = np.abs(target)
                scale = np.where(magnitude == 0, 1.0, magnitude)
                scale = float(scale) if scale.ndim == 0 else scale
            elif np.any(scale == 0):
                raise ValueError("scale must be nonzero.")

            residual = (
                (value - target) / scale if isinstance(value, Node)
                else lambda ctx: (value(ctx) - target) / scale
            )
            objectives = (Objective(residual, 1.0 if weight is None else weight, label),)
        elif target is not None or scale is not None:
            raise ValueError("target and scale require value.")
        elif label is not None or weight is not None:
            if len(objectives) != 1 or isinstance(objectives[0], Objective):
                raise ValueError("label and weight require a single residual Node or callable.")
            objectives = (Objective(objectives[0], 1.0 if weight is None else weight, label),)

        self._objectives.extend([normalize_to_objective(objective) for objective in objectives])
        return self

    def require(self, *constraints, label=None):
        """Add hard optimization constraints, on top of the Simulation's own contstraints.

        Parameters
        ----------
        *constraints : Comparison, Constraint or callable
            Comparisons such as ``sim[lens].after.w <= 1 * mm``, `Constraint` objects,
            or callables without arguments that return a comparison.
        label : str, optional
            Name for a single comparison, used in reports.

        Returns
        -------
        Solver
            ``self``, so calls can be chained.

        Raises
        ------
        RuntimeError
            If the model is already fixed by a previous :meth:`solve`.
        ValueError
            If `label` is given with several comparisons or with a `Constraint`.

        Notes
        -----
        A deferred callable must be differentiable by the numerical backend;
        diffractix does not check this.
        """
        if self._cache is not None:
            raise RuntimeError("The optimization model is fixed; create a new Solver.")
        constraints = label_constraints(constraints, label)
        self._constraints.extend(tuple(normalize_to_constraint(item) for item in constraints))
        return self

    def solve(
        self,
        *,
        optimizer: Optimizer | str = Optimizer.SCIPY,
        method=None,
        options: dict | None = None,
        feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
        jit: bool = True,
        x0=None,
        x_lower=None,
        x_upper=None,
    ) -> Solution:
        """Solve the optimization problem.

        Parameters
        ----------
        optimizer : Optimizer or str, optional
            Backend to use. Defaults to ``Optimizer.SCIPY``.
        method : str, optional
            Backend-specific algorithm name. Defaults to the backend's own default.
        options : dict, optional
            Backend-specific options. Defaults to ``None``.
        feasibility_tolerance : float, optional
            Maximum constraint or bound violation still counted as feasible. Defaults
            to `DEFAULT_FEASIBILITY_TOLERANCE`.
        jit : bool, optional
            Whether to just-in-time compile the objective and constraints. Defaults
            to ``True``; pass ``False`` if a deferred callable's control flow cannot
            be traced.
        x0 : array_like, optional
            Starting point, in `simulation.parameter_data` order. Defaults to the
            Simulation's initial values.
        x_lower, x_upper : array_like, optional
            Parameter bounds, in `simulation.parameter_data` order. Default to the
            Simulation's own bounds.

        Returns
        -------
        Solution
            The solved values and model evaluations.

        Raises
        ------
        ValueError
            If `optimizer` is not a supported name, `feasibility_tolerance` is not
            finite or is negative, the model has no variable Parameter, the
            model has no objective or constraint, `x0`, `x_lower` or `x_upper` has
            the wrong shape, `x0` is non-finite, `x_lower` or `x_upper` is NaN,
            `x_lower` exceeds `x_upper`, or `x0` lies outside the bounds.
        TypeError
            If `feasibility_tolerance` or `jit` has the wrong type, or the model
            contains something other than `Objective` or `Constraint` instances.

        Notes
        -----
        The first `solve` fixes the model; later calls reuse its compiled program and
        may use different `x0`, bounds or `optimizer`.
        """
        n = len(self.simulation._optics_program.parameter_indices)
        try:
            optimizer = Optimizer(optimizer.lower() if isinstance(optimizer, str) else optimizer)
        except ValueError:
            choices = ", ".join(repr(item.value) for item in Optimizer)
            raise ValueError(f"Unsupported optimizer {optimizer!r}; choose one of {choices}.") from None
        if n == 0:
            raise ValueError("Optimization problem must have at least one variable parameter.")
        if isinstance(feasibility_tolerance, bool) or not isinstance(feasibility_tolerance, Real):
            raise TypeError("feasibility_tolerance must be a real numeric scalar.")
        if not isfinite(feasibility_tolerance) or feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and non-negative.")
        if not isinstance(jit, (bool, np.bool_)):
            raise TypeError("jit must be a bool.")
        if not all(isinstance(item, Objective) for item in self._objectives):
            raise TypeError("Solver objectives must contain only Objective instances.")
        if not all(isinstance(item, Constraint) for item in self._constraints):
            raise TypeError("Solver constraints must contain only Constraint instances.")
        if not (
            self._objectives or self._constraints
            or self.simulation._constraint_descriptors or self.simulation._callable_constraints
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


        jit = bool(jit)
        if self._cache is None or self._cache_jit != jit:
            with _explain_tracing_errors(jit):
                self._cache = self._build_cache(jit, initial_values)
            self._cache_jit = jit

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

        match optimizer:
            case Optimizer.IPOPT: solve = solve_ipopt
            case Optimizer.SCIPY: solve = solve_scipy
            case Optimizer.NLOPT: solve = solve_nlopt

        scaled_problem = problem.scale()
        with _explain_tracing_errors(jit):
            result = solve(scaled_problem, method=method, options={} if options is None else options)
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
        tracer = SimulationResultTracer(self.simulation.element_data, self.simulation._optical_inputs[0].initial_state)

        (
            solver_constraint_program, 
            solver_constraint_descriptors, 
            callable_solver_constraints
        ) = prepare_constraints(self._constraints, tracer=tracer, bindings=self.simulation.bindings)

        system_constraint_program = self.simulation._constraint_program
        system_constraint_descriptors = self.simulation._constraint_descriptors
        callable_system_constraints = self.simulation._callable_constraints

        # make both programs use the same parameter and runtime Symbol indices
        parameter_indices = self.simulation._optics_program.parameter_indices
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
        tracer = SimulationResultTracer(self.simulation.element_data, self.simulation._optical_inputs[0].initial_state)
        program, descriptors, callables = prepare_objectives(
            self._objectives, tracer=tracer, bindings=self.simulation.bindings,
        )
        program = remap_parameters(program, self.simulation._optics_program.parameter_indices)
        return program, descriptors, callables


    def _build_cache(self, jit, initial_values):
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
            jit=jit,
        )

        return self._SolverCache(
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
        initial_values, jit,
    ):
        def evaluate_callable(entry, view):
            value = entry.quantity(view)
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

        full_objective = backend_jit(objective) if jit else objective
        full_constraints = backend_jit(constraints) if jit else constraints

        def checked_objective(theta):
            return check_finite_value(full_objective(theta), "Objective cost")

        def checked_constraints(theta):
            return check_finite_value(full_constraints(theta), "Constraint vector")

        callbacks = self._Callbacks(
            objective          = checked_objective,
            gradient           = backend_jit(objective_gradient)  if jit else objective_gradient,
            objective_hessian  = backend_jit(objective_hessian)   if jit else objective_hessian,
            constraints        = checked_constraints,
            jacobian           = backend_jit(constraint_jacobian) if jit else constraint_jacobian,
            constraint_hessian = backend_jit(constraint_hessian)  if jit else constraint_hessian,
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

        cache.reduced_callbacks = self._Callbacks(
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
