from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import autograd.numpy as np

from .ast import (
    Node,
    Literal,
    Parameter,
    SystemVar,
    InputNode,
    BinaryOp,
    UnaryOp,
)
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


# --------------------
# PROGRAM INSTRUCTIONS
# --------------------

@dataclass(frozen=True)
class ConstInstruction:
    value: float                  # compiled constant value
    opcode = Opcode.CONST


@dataclass(frozen=True)
class VariableInstruction:
    index: int                    # index into supplied variable values
    opcode = Opcode.VARIABLE


@dataclass(frozen=True)
class BinaryInstruction:
    op: Op                        # semantic operation
    func: Callable                # pre-bound numerical implementation
    left_index: int               # index of left input value
    right_index: int              # index of right input value
    opcode = Opcode.BINARY


@dataclass(frozen=True)
class UnaryInstruction:
    op: Op                        # semantic operation
    func: Callable                # pre-bound numerical implementation
    operand_index: int            # index of input value
    opcode = Opcode.UNARY


Instruction = (
    ConstInstruction
    | VariableInstruction
    | BinaryInstruction
    | UnaryInstruction
)


@dataclass(frozen=True)
class ASTProgram:
    """
    Linear numerical program generated from an AST.

    Each instruction produces exactly one value. The index of the instruction
    is also the index of its result in the evaluation value array.
    """
    instructions: tuple[Instruction, ...]  # ordered numerical instructions
    root_indices: tuple[int, ...]          # value indices returned as outputs


@dataclass(frozen=True)
class CompiledAST:
    """Optimized, executable representation of an AST."""

    program: ASTProgram                  # optimized numerical program
    variables: tuple[Parameter, ...]     # ordered variable Parameters
    initial_values: np.ndarray           # initial variable values

    def evaluate(self, variable_values: np.ndarray) -> np.ndarray:
        """Evaluate the compiled AST for the supplied variable values."""
        if len(variable_values) != len(self.variables):
            raise ValueError(
                f"Expected {len(self.variables)} variable values, "
                f"got {len(variable_values)}."
            )

        values = []

        for instruction in self.program.instructions:
            if instruction.opcode is Opcode.CONST:
                values.append(instruction.value)

            elif instruction.opcode is Opcode.VARIABLE:
                values.append(variable_values[instruction.index])

            elif instruction.opcode is Opcode.BINARY:
                left = values[instruction.left_index]
                right = values[instruction.right_index]
                values.append(instruction.func(left, right))

            elif instruction.opcode is Opcode.UNARY:
                operand = values[instruction.operand_index]
                values.append(instruction.func(operand))

        return np.array([
            values[index]
            for index in self.program.root_indices
        ])

    def __call__(self, variable_values: np.ndarray) -> np.ndarray:
        return self.evaluate(variable_values)


# --------------------
# PROGRAM OPTIMIZATION
# --------------------

def constant_key(value) -> tuple:
    """
    Return a stable key for constant deduplication.

    float.hex() preserves distinctions such as +0.0 and -0.0.
    """
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        return float, value.hex()

    if isinstance(value, complex):
        return complex, value.real.hex(), value.imag.hex()

    return type(value), value


def scalar_value(value):
    """Convert NumPy scalar results from constant folding to Python scalars."""
    return value.item() if hasattr(value, "item") else value


def optimize_program(program: ASTProgram) -> ASTProgram:
    """
    Optimize a compiled numerical program without changing its semantics.

    The current passes perform constant folding, common-subexpression
    elimination, and removal of values that no longer contribute to an output.
    """
    program = fold_constants_and_cse(program)
    program = eliminate_dead_code(program)
    return program


def fold_constants_and_cse(program: ASTProgram) -> ASTProgram:
    """
    Fold constant expressions and merge equivalent compiled instructions.

    CSE happens here rather than during AST construction. Distinct Parameter
    objects therefore remain independent even if their metadata is identical.
    """
    optimized: list[Instruction] = []

    # Map each value produced by the original program to its corresponding
    # value index in the optimized program.
    old_to_new_index: dict[int, int] = {}

    # Maps an instruction's semantic identity to an existing result value.
    canonical_indices: dict[tuple, int] = {}

    # Known compile-time values, indexed by optimized value index.
    constant_values: dict[int, float] = {}

    def intern(instruction: Instruction, key: tuple) -> int:
        """Reuse an equivalent instruction or append a new one."""
        if key in canonical_indices:
            return canonical_indices[key]

        value_index = len(optimized)
        optimized.append(instruction)
        canonical_indices[key] = value_index

        if instruction.opcode is Opcode.CONST:
            constant_values[value_index] = instruction.value

        return value_index

    for old_index, instruction in enumerate(program.instructions):
        if instruction.opcode is Opcode.CONST:
            instruction = ConstInstruction(instruction.value)
            key = (
                Opcode.CONST,
                constant_key(instruction.value),
            )

        elif instruction.opcode is Opcode.VARIABLE:
            key = (
                Opcode.VARIABLE,
                instruction.index,
            )

        elif instruction.opcode is Opcode.UNARY:
            operand_index = old_to_new_index[instruction.operand_index]

            if operand_index in constant_values:
                value = scalar_value(
                    instruction.func(
                        constant_values[operand_index]
                    )
                )

                instruction = ConstInstruction(value)
                key = (
                    Opcode.CONST,
                    constant_key(value),
                )

            else:
                instruction = UnaryInstruction(
                    op=instruction.op,
                    func=instruction.func,
                    operand_index=operand_index,
                )

                key = (
                    Opcode.UNARY,
                    instruction.op,
                    operand_index,
                )

        elif instruction.opcode is Opcode.BINARY:
            left_index = old_to_new_index[instruction.left_index]
            right_index = old_to_new_index[instruction.right_index]

            if (
                left_index in constant_values
                and right_index in constant_values
            ):
                value = scalar_value(
                    instruction.func(
                        constant_values[left_index],
                        constant_values[right_index],
                    )
                )

                instruction = ConstInstruction(value)
                key = (
                    Opcode.CONST,
                    constant_key(value),
                )

            else:
                instruction = BinaryInstruction(
                    op=instruction.op,
                    func=instruction.func,
                    left_index=left_index,
                    right_index=right_index,
                )

                key = (
                    Opcode.BINARY,
                    instruction.op,
                    left_index,
                    right_index,
                )

        else:
            raise RuntimeError(
                f"Unknown instruction type: {type(instruction).__name__}"
            )

        old_to_new_index[old_index] = intern(instruction, key)

    root_indices = tuple(
        old_to_new_index[index]
        for index in program.root_indices
    )

    return ASTProgram(
        instructions=tuple(optimized),
        root_indices=root_indices,
    )


def instruction_dependencies(
    instruction: Instruction,
) -> tuple[int, ...]:
    """Return the value indices required by an instruction."""
    if instruction.opcode is Opcode.BINARY:
        return (
            instruction.left_index,
            instruction.right_index,
        )

    if instruction.opcode is Opcode.UNARY:
        return (instruction.operand_index,)

    return ()


def remap_instruction(instruction: Instruction, index_map: dict[int, int]) -> Instruction:
    """Rewrite an instruction to reference a new value-index layout."""
    if instruction.opcode is Opcode.BINARY:
        return BinaryInstruction(
            op=instruction.op,
            func=instruction.func,
            left_index=index_map[instruction.left_index],
            right_index=index_map[instruction.right_index],
        )

    if instruction.opcode is Opcode.UNARY:
        return UnaryInstruction(
            op=instruction.op,
            func=instruction.func,
            operand_index=index_map[instruction.operand_index],
        )

    return instruction


def eliminate_dead_code(program: ASTProgram) -> ASTProgram:
    """
    Remove instructions that do not contribute to any requested root.

    Constant folding can make previously required intermediate values
    unreachable, so this pass also compacts the value indices afterwards.
    """
    live_indices = set(program.root_indices)
    pending = list(program.root_indices)

    # Walk backwards from outputs to determine which values are still needed.
    while pending:
        value_index = pending.pop()
        instruction = program.instructions[value_index]

        for dependency_index in instruction_dependencies(instruction):
            if dependency_index not in live_indices:
                live_indices.add(dependency_index)
                pending.append(dependency_index)

    ordered_indices = sorted(live_indices)

    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(ordered_indices)
    }

    instructions = tuple(
        remap_instruction(
            program.instructions[old_index],
            index_map,
        )
        for old_index in ordered_indices
    )

    root_indices = tuple(
        index_map[index]
        for index in program.root_indices
    )

    return ASTProgram(
        instructions=instructions,
        root_indices=root_indices,
    )


# -----------
# COMPILATION
# -----------
def compile_ast(roots: Sequence[Node], context: ASTContext | None = None) -> CompiledAST:
    """
    Compile an AST into an optimized differentiable numerical program.

    Compilation snapshots the current graph structure, fixed Parameter values,
    and context values. Evaluation afterwards depends only on the supplied
    variable values.
    """
    roots = tuple(roots)
    context = dict(context or {})

    variables = collect_variables(roots, context)
    variable_indices = {
        parameter: index
        for index, parameter in enumerate(variables)
    }

    initial_values = np.array(
        [parameter.value for parameter in variables],
        dtype=float,
    )

    # Each AST node maps to the index of the value produced for that node.
    value_indices: dict[Node, int] = {}

    # Tracks the current recursion path so cycles can be detected explicitly.
    active: set[Node] = set()

    instructions: list[Instruction] = []

    def emit(instruction: Instruction) -> int:
        """Append an instruction and return the index of its result value."""
        value_index = len(instructions)
        instructions.append(instruction)
        return value_index

    def compile_node(node: Node) -> int:
        """Compile a node and return the index of its resulting value."""
        if node in active:
            raise ASTCycleError(
                f"Cycle detected at {describe_node(node)}."
            )

        # Explicitly shared AST nodes reuse the same compiled value.
        if node in value_indices:
            return value_indices[node]

        active.add(node)

        if isinstance(node, Literal):
            value_index = emit(
                ConstInstruction(node.value)
            )

        elif isinstance(node, Parameter):
            if node.is_variable:
                value_index = emit(
                    VariableInstruction(
                        variable_indices[node]
                    )
                )
            else:
                value_index = emit(
                    ConstInstruction(node.value)
                )

        elif isinstance(node, BinaryOp):
            left_index = compile_node(node.left)
            right_index = compile_node(node.right)

            value_index = emit(
                BinaryInstruction(
                    op=node.op,
                    func=node.op.func,
                    left_index=left_index,
                    right_index=right_index,
                )
            )

        elif isinstance(node, UnaryOp):
            operand_index = compile_node(node.operand)

            value_index = emit(
                UnaryInstruction(
                    op=node.op,
                    func=node.op.func,
                    operand_index=operand_index,
                )
            )

        elif isinstance(node, InputNode):
            if node.node is None:
                raise UnresolvedInputError(
                    "Encountered an empty InputNode."
                )

            # InputNodes are compile-time indirections and produce no
            # instruction of their own.
            value_index = compile_node(node.node)

        elif isinstance(node, SystemVar):
            value = resolve_system_var(node, context)

            # SystemVars likewise disappear during compilation once resolved.
            if isinstance(value, Node):
                value_index = compile_node(value)
            else:
                value_index = emit(
                    ConstInstruction(value)
                )

        else:
            raise UnsupportedNodeError(
                f"Unsupported AST node type: {type(node).__name__}"
            )

        active.remove(node)
        value_indices[node] = value_index

        return value_index

    root_indices = tuple(
        compile_node(root)
        for root in roots
    )

    raw_program = ASTProgram(
        instructions=tuple(instructions),
        root_indices=root_indices,
    )

    optimized_program = optimize_program(raw_program)

    return CompiledAST(
        program=optimized_program,
        variables=variables,
        initial_values=initial_values,
    )