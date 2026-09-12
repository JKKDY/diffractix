from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable
from collections.abc import Hashable, Mapping, Sequence

import autograd.numpy as np

from .node import (
    Node,
    Literal,
    Parameter,
    Symbol,
    InputNode,
    BinaryOp,
    UnaryOp,
    Scalar,
)
from .ops import Op





ASTContext = Mapping[Hashable, Scalar | Node]
ParameterSnapshot = Mapping[int, object]


class ASTError(Exception):
    """Base exception for AST traversal and compilation errors."""


class ASTCycleError(ASTError):
    """Raised when a cycle is found in the AST."""


class UnresolvedInputError(ASTError):
    """Raised when an empty InputNode is encountered."""


class UnsupportedNodeError(ASTError):
    """Raised when an unknown Node subclass is encountered."""



def describe_node(node: Node) -> str:
    """Return a non-recursive description of an AST node."""
    return f"{type(node).__name__}(id={id(node)})"


def parameter_state(
    parameter: Parameter,
    parameter_snapshot: ParameterSnapshot | None = None,
) -> tuple[Scalar, bool]:
    """Return the authoritative value and variable state for a Parameter."""
    parameter_id = id(parameter)

    if parameter_snapshot is not None and parameter_id in parameter_snapshot:
        info = parameter_snapshot[parameter_id]
        return info.value, info.parameter_index is not None

    return parameter.value, parameter.is_variable


def iter_children(node: Node, context: ASTContext) -> tuple[Node, ...]:
    """
    Return the direct dependencies of a node.

    Symbols bound through the context expose the bound node as their child.
    Unbound Symbols and scalar context values are terminal.
    """
    if isinstance(node, (Literal, Parameter)):
        return ()

    if isinstance(node, BinaryOp):
        return node.left, node.right

    if isinstance(node, UnaryOp):
        return (node.operand,)

    if isinstance(node, InputNode):
        if node.node is None:
            raise UnresolvedInputError("Encountered an empty InputNode.")
        return (node.node,)

    if isinstance(node, Symbol):
        if node.key not in context:
            return ()

        value = context[node.key]
        return (value,) if isinstance(value, Node) else ()

    raise UnsupportedNodeError(
        f"Unsupported AST node type: {type(node).__name__}"
    )


def walk_ast(roots: Sequence[Node], context: ASTContext | None = None):
    """
    Walk the reachable AST once in depth-first order.

    Nodes are deduplicated by object identity. Explicitly shared graph nodes
    are therefore visited only once.
    """
    context = {} if context is None else context
    seen: set[int] = set()
    active: set[int] = set()

    def visit(node: Node):
        node_id = id(node)

        if node_id in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")

        if node_id in seen:
            return

        seen.add(node_id)
        active.add(node_id)

        yield node

        for child in iter_children(node, context):
            yield from visit(child)

        active.remove(node_id)

    for root in roots:
        if not isinstance(root, Node):
            raise TypeError(f"AST root must be a Node, got {type(root).__name__}.")
        yield from visit(root)



def collect_parameters(roots: Sequence[Node], context: ASTContext | None = None) -> tuple[Parameter, ...]:
    """Collect all unique Parameter leaves reachable from roots."""
    return tuple(
        node for node in walk_ast(roots, context) if isinstance(node, Parameter)
    )


def collect_variables(
    roots: Sequence[Node],
    context: ASTContext | None = None,
    parameter_snapshot: ParameterSnapshot | None = None,
) -> tuple[Parameter, ...]:
    """
    Collect unique variable Parameters reachable from the roots.

    Parameters are deduplicated by object identity and returned in deterministic
    first-encounter order.
    """
    return tuple(
        parameter for parameter in collect_parameters(roots, context)
        if parameter_state(parameter, parameter_snapshot)[1]
    )


def clone_ast(roots: Sequence[Node], *, preserve_owners: bool = True) -> tuple[Node, ...]:
    """
    Clone an AST while preserving its sharing structure.

    Two references to the same source node will reference the same cloned node.
    Structurally equivalent but distinct source nodes remain distinct.

    Symbols are copied but not resolved.
    """
    memo: dict[int, Node] = {}
    active: set[int] = set()

    def clone(node: Node) -> Node:
        node_id = id(node)

        if node_id in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")

        if node_id in memo:
            return memo[node_id]

        active.add(node_id)

        if isinstance(node, Literal):
            result = Literal(node.value)

        elif isinstance(node, Parameter):
            result = Parameter(
                value=node.value,
                name=node.name,
                variable=node.is_variable,
                lower_bound=node.lower_bound,
                upper_bound=node.upper_bound,
                owner=node.owner if preserve_owners else None,
            )

        elif isinstance(node, BinaryOp):
            result = BinaryOp(
                node.op,
                clone(node.left),
                clone(node.right),
            )

        elif isinstance(node, UnaryOp):
            result = UnaryOp(
                node.op,
                clone(node.operand),
            )

        elif isinstance(node, InputNode):
            result = InputNode(
                clone(node.node) if node.node is not None else None
            )

        elif isinstance(node, Symbol):
            result = Symbol(node.key)

        else:
            raise UnsupportedNodeError(f"Unsupported AST node type: {type(node).__name__}")

        active.remove(node_id)
        memo[node_id] = result

        return result

    return tuple(clone(root) for root in roots)



