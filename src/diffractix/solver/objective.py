from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from diffractix.graph import Node, Symbol, SymbolicControlFlowError, precompile, walk_ast
from .utils import callable_arity


@dataclass(frozen=True)
class Objective:
    """A soft optimization objective: one or more residuals, weighted in the total cost.

    Pass it to :meth:`Solver.target`. The total cost is the sum, over every
    objective, of ``weight * sum(residual**2)``.

    Parameters
    ----------
    quantity : Node or callable
        The residual: a graph expression, or a callable accepting the result
        (traceable, or deferred if it branches on result values).
    weight : float, optional
        Multiplies the squared residual in the total cost. Defaults to 1.
    label : str, optional
        Name used in reports.

    Raises
    ------
    TypeError
        If `quantity` is not a Node or callable.
    ValueError
        If `weight` is not finite, or is negative.

    Notes
    -----
    A deferred callable must be differentiable by the numerical backend;
    diffractix does not check this.
    """
    quantity: Node | Callable
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.quantity, Node) and not callable(self.quantity):
            raise TypeError("quantity must be a Node or callable.")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and >= 0.")


@dataclass(frozen=True)
class ObjectiveDescriptor:
    """Weight and declaration position for one objective program output."""

    position: int
    weight: float
    label: str | None


def prepare_objectives(objectives, tracer, bindings):
    """Trace and precompile residuals, retaining dynamic objective callables."""
    from diffractix.simulation.result import ResultSymbolKey

    roots = []
    descriptors = []
    callables = []
    for position, entry in enumerate(objectives):
        objective = normalize_to_objective(entry)
        expression = objective.quantity
        if not isinstance(expression, Node):
            try:
                expression = expression(tracer)
            except SymbolicControlFlowError:
                callables.append((position, objective))
                continue
            if not isinstance(expression, Node):
                # Array residuals remain callables because graph literals are scalars.
                if np.ndim(expression) != 0:
                    callables.append((position, objective))
                    continue
                expression = Node._make_literal(expression)
        # only known symbols allowed
        for node in walk_ast((expression,), bindings):
            if isinstance(node, Symbol) and node.key not in bindings and not isinstance(node.key, ResultSymbolKey):
                raise ValueError(f"Objective {position} has an unresolved Symbol binding: {node.key!r}.")
        roots.append(expression)
        descriptors.append(ObjectiveDescriptor(position, objective.weight, objective.label))

    return precompile(roots, bindings), tuple(descriptors), tuple(callables)



def normalize_to_objective(value) -> Objective:
    if isinstance(value, Node):
        return Objective(value)

    objective = value if isinstance(value, Objective) else Objective(value)
    if isinstance(objective.quantity, Node) or callable_arity(objective.quantity) == 1:
        return objective

    result = objective.quantity()
    if not isinstance(result, Node):
        raise TypeError("A zero-argument objective callable must return a Node.")
    return Objective(result, objective.weight, objective.label)
