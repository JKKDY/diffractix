from collections.abc import Mapping, Sequence

from .node import (
    Node,
    Literal,
    Parameter,
    Symbol,
    InputNode,
    BinaryOp,
    UnaryOp,
)

from .errors import ASTCycleError, UnresolvedInputError, UnsupportedNodeError



def describe_node(node: Node) -> str:
    """Return a non-recursive description of an AST node."""
    return f"{type(node).__name__}(id={id(node)})"


def evaluate_ast(root: Node, bindings: Mapping = {}, parameter_values: Mapping | None = None):
    """Evaluate a graph using symbol bindings and optional parameter values.

    When parameter_values is supplied, it must contain every Parameter in the
    graph, keyed by its identity. Otherwise current Parameter values are used.
    Symbol bindings may contain numerical values or other Nodes.
    """
    active = set()
    values = {}

    def evaluate(node):
        node_id = id(node)
        if node_id in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")
        if node_id in values:
            return values[node_id]

        active.add(node_id)
        try:
            if isinstance(node, Literal):
                value = node.value
            
            elif isinstance(node, Parameter):
                if parameter_values is None:
                    value = node.value
                elif node_id in parameter_values:
                    value = parameter_values[node_id]
                else:
                    raise KeyError(f"Missing value for Parameter {node!r} (id {node_id}).") from None
            
            elif isinstance(node, Symbol):
                if node.key in bindings:
                    value = bindings[node.key]
                else:
                    raise KeyError(f"Missing binding for Symbol key {node.key!r}.") from None
                if isinstance(value, Node):
                    value = evaluate(value)
            
            elif isinstance(node, InputNode):
                if node.node is None:
                    raise UnresolvedInputError("Encountered an empty InputNode.")
                value = evaluate(node.node)
            
            elif isinstance(node, UnaryOp):
                value = node.op.func(evaluate(node.operand))
            
            elif isinstance(node, BinaryOp):
                value = node.op.func(evaluate(node.left), evaluate(node.right))
            
            else:
                raise UnsupportedNodeError(f"Unsupported AST node type: {type(node).__name__}")
        finally:
            active.remove(node_id)

        values[node_id] = value
        return value

    return evaluate(root)


def clone_ast(node: Node, memo: dict[int, Node] | None = None) -> Node:
    """Copy the structure of a graph so later edits to the original do not reach it.

    Operations and InputNode handles are copied. Parameters, Literals and Symbols
    are shared, so Parameters keep their identity (and with it their theta
    association and live values). Shared subexpressions stay shared; pass one
    memo to several calls to also share them across the cloned roots.
    """
    if memo is None:
        memo = {}
    active = set()

    def clone(node):
        node_id = id(node)
        if node_id in memo:
            return memo[node_id]
        if node_id in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")
        active.add(node_id)

        if isinstance(node, (Literal, Parameter, Symbol)):
            copy = node
        elif isinstance(node, BinaryOp):
            copy = BinaryOp(node.op, clone(node.left), clone(node.right))
        elif isinstance(node, UnaryOp):
            copy = UnaryOp(node.op, clone(node.operand))
        elif isinstance(node, InputNode):
            if node.node is None:
                raise UnresolvedInputError("Encountered an empty InputNode.")
            copy = InputNode(clone(node.node))
        else:
            raise UnsupportedNodeError(f"Unsupported AST node type: {type(node).__name__}")

        active.remove(node_id)
        memo[node_id] = copy
        return copy

    if not isinstance(node, Node):
        raise TypeError(f"AST root must be a Node, got {type(node).__name__}.")
    return clone(node)


def walk_ast(roots: Sequence[Node], bindings:Mapping = {}):
    """
    Walk the reachable AST once in depth-first order.

    Nodes are deduplicated by object identity. Explicitly shared graph nodes
    are therefore visited only once.
    """
    seen = set()  # ids of seen nodes
    active = set()  # ids of notes currently being processed

    def get_children(node: Node, bindings) -> tuple[Node, ...]:
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
            if node.key not in bindings:
                return ()
            value = bindings[node.key]
            return (value,) if isinstance(value, Node) else () # bindings can be AST 

        raise UnsupportedNodeError(f"Unsupported AST node type: {type(node).__name__}")

    def visit(node: Node):
        node_id = id(node)

        if node_id in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")
        if node_id in seen:
            return

        seen.add(node_id)
        active.add(node_id)

        yield node
        for child in get_children(node, bindings):
            yield from visit(child)
        active.remove(node_id)

    for root in roots:
        if not isinstance(root, Node):
            raise TypeError(f"AST root must be a Node, got {type(root).__name__}.")
        yield from visit(root)
