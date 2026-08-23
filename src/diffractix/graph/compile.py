from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import autograd.numpy as np

from .ast import Node, Literal, Parameter, SystemVar, InputNode, BinaryOp, UnaryOp
from .ops import Op
from .utils import (
    ASTContext,
    ASTCycleError,
    UnresolvedInputError,
    UnsupportedNodeError,
    collect_variables,
    resolve_system_var,
    describe_node,
)


class Opcode(Enum):
    CONST = auto()
    VARIABLE = auto()
    BINARY = auto()
    UNARY = auto()



@dataclass(frozen=True)
class ConstInstruction:
    value: float                  # compiled constant value
    opcode = Opcode.CONST


@dataclass(frozen=True)
class VariableInstruction:
    index: int                    # index into variable values
    opcode = Opcode.VARIABLE


@dataclass(frozen=True)
class BinaryInstruction:
    op: Op                        # binary operation
    left_slot: int                # slot of left operand
    right_slot: int               # slot of right operand
    opcode = Opcode.BINARY


@dataclass(frozen=True)
class UnaryInstruction:
    op: Op                        # unary operation
    operand_slot: int             # slot of operand
    opcode = Opcode.UNARY


Instruction = (ConstInstruction | VariableInstruction | BinaryInstruction | UnaryInstruction)



@dataclass(frozen=True)
class CompiledAST:
    evaluate: Callable[[np.ndarray], np.ndarray]  # variable values -> root values
    variables: tuple[Parameter, ...]              # ordered variable parameters
    initial_values: np.ndarray                    # initial variable values

    def __call__(self, variable_values: np.ndarray) -> np.ndarray:
        return self.evaluate(variable_values)


def compile_ast(roots: Sequence[Node], context: ASTContext | None = None) -> CompiledAST:
    roots = tuple(roots)
    context = dict(context or {})

    variables = collect_variables(roots, context)
    variable_indices = {p: i for i, p in enumerate(variables)}
    initial_values = np.array([p.value for p in variables], dtype=float)

    slots: dict[Node, int] = {}       # compiled slot for each AST node
    active: set[Node] = set()         # nodes currently being compiled
    program: list[Instruction] = []   # linear instruction sequence

    def emit(instruction: Instruction) -> int:
        slot = len(program)
        program.append(instruction)
        return slot

    def compile_node(node: Node) -> int:
        if node in active:
            raise ASTCycleError(f"Cycle detected at {describe_node(node)}.")

        if node in slots:
            return slots[node]

        active.add(node)

        if isinstance(node, Literal):
            slot = emit(ConstInstruction(node.value))

        elif isinstance(node, Parameter):
            if node.is_variable:
                slot = emit(VariableInstruction(variable_indices[node]))
            else:
                slot = emit(ConstInstruction(node.value))

        elif isinstance(node, BinaryOp):
            left_slot = compile_node(node.left)
            right_slot = compile_node(node.right)
            slot = emit(BinaryInstruction(node.op, left_slot, right_slot))

        elif isinstance(node, UnaryOp):
            operand_slot = compile_node(node.operand)
            slot = emit(UnaryInstruction(node.op, operand_slot))

        elif isinstance(node, InputNode):
            if node.node is None:
                raise UnresolvedInputError("Encountered an empty InputNode.")
            slot = compile_node(node.node)

        elif isinstance(node, SystemVar):
            value = resolve_system_var(node, context)
            slot = compile_node(value) if isinstance(value, Node) else emit(ConstInstruction(value))

        else:
            raise UnsupportedNodeError(
                f"Unsupported AST node type: {type(node).__name__}"
            )

        active.remove(node)
        slots[node] = slot
        return slot

    root_slots = tuple(compile_node(root) for root in roots)
    program = tuple(program)

    def evaluate(variable_values: np.ndarray) -> np.ndarray:
        if len(variable_values) != len(variables):
            raise ValueError(
                f"Expected {len(variables)} variable values, "
                f"got {len(variable_values)}."
            )

        values = []

        for instruction in program:
            if instruction.opcode is Opcode.CONST:
                values.append(instruction.value)

            elif instruction.opcode is Opcode.VARIABLE:
                values.append(variable_values[instruction.index])

            elif instruction.opcode is Opcode.BINARY:
                left = values[instruction.left_slot]
                right = values[instruction.right_slot]
                values.append(instruction.op.func(left, right))

            elif instruction.opcode is Opcode.UNARY:
                operand = values[instruction.operand_slot]
                values.append(instruction.op.func(operand))

        return np.array([values[slot] for slot in root_slots])

    return CompiledAST(
        evaluate=evaluate,
        variables=variables,
        initial_values=initial_values,
    )