from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import autograd.numpy as np

from .ast import Node, Literal, Parameter, SystemVar, InputNode, BinaryOp, UnaryOp
from .ops import Op



ASTContext = Mapping[str, Scalar | Node]


class ASTError(Exception):
    """Base exception for AST traversal and compilation errors."""


class ASTCycleError(ASTError):
    """Raised when a cycle is found in the AST."""


class UnresolvedSystemVarError(ASTError):
    """Raised when a SystemVar cannot be resolved from the context."""


class UnresolvedInputError(ASTError):
    """Raised when an empty InputNode is encountered."""


class UnsupportedNodeError(ASTError):
    """Raised when an unknown Node subclass is encountered."""




def resolve_system_var(node: SystemVar, context: ASTContext) -> Scalar | Node:
    """Resolve a SystemVar against the supplied context."""
    try:
        return context[node.name]
    except KeyError:
        raise UnresolvedSystemVarError(
            f"No value provided for SystemVar {node.name!r}."
        ) from None


def iter_children(node: Node, context: ASTContext) -> tuple[Node, ...]:
    """
    Return the direct dependencies of a node.

    SystemVars are resolved through the context. Scalar context values are
    terminal and therefore do not appear as children.
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

    if isinstance(node, SystemVar):
        value = resolve_system_var(node, context)
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
    seen: set[Node] = set()
    active: set[Node] = set()

    def visit(node: Node):
        if node in active:
            raise ASTCycleError(f"Cycle detected at {node!r}.")

        if node in seen:
            return

        seen.add(node)
        active.add(node)

        yield node

        for child in iter_children(node, context):
            yield from visit(child)

        active.remove(node)

    for root in roots:
        if not isinstance(root, Node):
            raise TypeError(
                f"AST root must be a Node, got {type(root).__name__}."
            )

        yield from visit(root)


def collect_variables(roots: Sequence[Node], context: ASTContext | None = None) -> tuple[Parameter, ...]:
    """
    Collect unique variable Parameters reachable from the roots.

    Parameters are deduplicated by object identity and returned in deterministic
    first-encounter order.
    """
    variables = []

    for node in walk_ast(roots, context):
        if isinstance(node, Parameter) and node.is_variable:
            variables.append(node)

    return tuple(variables)


def clone_ast(roots: Sequence[Node], *, preserve_owners: bool = True) -> tuple[Node, ...]:
    """
    Clone an AST while preserving its sharing structure.

    Two references to the same source node will reference the same cloned node.
    Structurally equivalent but distinct source nodes remain distinct.

    SystemVars are copied but not resolved.
    """
    memo: dict[Node, Node] = {}
    active: set[Node] = set()

    def clone(node: Node) -> Node:
        if node in active:
            raise ASTCycleError(f"Cycle detected at {node!r}.")

        if node in memo:
            return memo[node]

        active.add(node)

        if isinstance(node, Literal):
            result = Literal(node.value)

        elif isinstance(node, Parameter):
            result = Parameter(
                value=node.value,
                name=node.name,
                variable=node.is_variable,
                min_val=node.min_val,
                max_val=node.max_val,
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

        elif isinstance(node, SystemVar):
            result = SystemVar(node.name)

        else:
            raise UnsupportedNodeError(
                f"Unsupported AST node type: {type(node).__name__}"
            )

        active.remove(node)
        memo[node] = result
        return result

    return tuple(clone(root) for root in roots)








class _Opcode(Enum):
    CONST = auto()
    THETA = auto()
    BINARY = auto()
    UNARY = auto()


@dataclass(frozen=True)
class _ConstInstruction:
    value: float
    opcode = _Opcode.CONST


@dataclass(frozen=True)
class _ThetaInstruction:
    index: int
    opcode = _Opcode.THETA


@dataclass(frozen=True)
class _BinaryInstruction:
    op: Op
    left: int
    right: int
    opcode = _Opcode.BINARY


@dataclass(frozen=True)
class _UnaryInstruction:
    op: Op
    operand: int
    opcode = _Opcode.UNARY


_Instruction = (
    _ConstInstruction
    | _ThetaInstruction
    | _BinaryInstruction
    | _UnaryInstruction
)


@dataclass(frozen=True)
class CompiledAST:
    transform: Callable[[np.ndarray], np.ndarray]
    variables: tuple[Parameter, ...]
    initial_values: np.ndarray

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.transform(theta)


def compile_ast(
    roots: list[Node] | tuple[Node, ...],
    context: ASTContext | None = None,
) -> CompiledAST:
    """Compile an AST into a pure differentiable theta -> roots transform."""

    roots = tuple(roots)
    context = dict(context or {})

    variables = collect_variables(roots, context)
    variable_indices = {parameter: i for i, parameter in enumerate(variables)}
    initial_values = np.array([p.value for p in variables], dtype=float)

    slots: dict[Node, int] = {}
    active: set[Node] = set()
    program: list[_Instruction] = []

    def emit(instruction: _Instruction) -> int:
        slot = len(program)
        program.append(instruction)
        return slot

    def compile_node(node: Node) -> int:
        if node in active:
            raise ASTCycleError(f"Cycle detected at {node!r}.")

        if node in slots:
            return slots[node]

        active.add(node)

        if isinstance(node, Literal):
            slot = emit(_ConstInstruction(node.value))

        elif isinstance(node, Parameter):
            if node.is_variable:
                slot = emit(_ThetaInstruction(variable_indices[node]))
            else:
                slot = emit(_ConstInstruction(node.value))

        elif isinstance(node, BinaryOp):
            left = compile_node(node.left)
            right = compile_node(node.right)
            slot = emit(_BinaryInstruction(node.op, left, right))

        elif isinstance(node, UnaryOp):
            operand = compile_node(node.operand)
            slot = emit(_UnaryInstruction(node.op, operand))

        elif isinstance(node, InputNode):
            if node.node is None:
                raise UnresolvedInputError("Encountered an empty InputNode.")
            slot = compile_node(node.node)

        elif isinstance(node, SystemVar):
            value = resolve_system_var(node, context)
            if isinstance(value, Node):
                slot = compile_node(value)
            else:
                slot = emit(_ConstInstruction(value))

        else:
            raise UnsupportedNodeError(
                f"Unsupported AST node type: {type(node).__name__}"
            )

        active.remove(node)
        slots[node] = slot
        return slot

    root_slots = tuple(compile_node(root) for root in roots)
    program = tuple(program)

    def transform(theta: np.ndarray) -> np.ndarray:
        if len(theta) != len(variables):
            raise ValueError(
                f"Expected theta of length {len(variables)}, got {len(theta)}."
            )

        values = []

        for instruction in program:
            if instruction.opcode is _Opcode.CONST:
                values.append(instruction.value)

            elif instruction.opcode is _Opcode.THETA:
                values.append(theta[instruction.index])

            elif instruction.opcode is _Opcode.BINARY:
                left = values[instruction.left]
                right = values[instruction.right]
                values.append(instruction.op.func(left, right))

            elif instruction.opcode is _Opcode.UNARY:
                operand = values[instruction.operand]
                values.append(instruction.op.func(operand))

        return np.array([values[slot] for slot in root_slots])

    return CompiledAST(
        transform=transform,
        variables=variables,
        initial_values=initial_values,
    )