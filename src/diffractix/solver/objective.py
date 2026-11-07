from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from diffractix.graph import Node, Symbol, SymbolicControlFlowError, precompile, walk_ast
from .utils import callable_arity


@dataclass(frozen=True)
class Objective:
    """Soft optimization objective producing one or more residuals."""
    evaluate: Node | Callable
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self):
        if not isinstance(self.evaluate, Node) and not callable(self.evaluate):
            raise TypeError("evaluate must be a Node or callable.")
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
        expression = objective.evaluate
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
    if isinstance(objective.evaluate, Node) or callable_arity(objective.evaluate) == 1:
        return objective

    result = objective.evaluate()
    if not isinstance(result, Node):
        raise TypeError("A zero-argument objective callable must return a Node.")
    return Objective(result, objective.weight, objective.label)
