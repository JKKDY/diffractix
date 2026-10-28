from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Hashable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from diffractix.backend import xp


from .node import (
    Node,
    Literal,
    Parameter,
    Symbol,
    InputNode,
    BinaryOp,
    UnaryOp
)

from .program import (
    Input, 
    Register, 
    Constant, 
    Operand,
    UnaryInstruction, 
    BinaryInstruction, 
    Program
)

from .errors import UnresolvedInputError, UnsupportedNodeError
from .graph_utils import walk_ast

from .ops import Op



def compile_graph(roots: Sequence[Node], bindings: Mapping = {}) -> Program:
    """Compile expression graphs into an optimized executable program."""
    return optimize(precompile_graph(roots, bindings))

def optimize(program: Program) -> Program:
    """Apply graph optimizations without reusing registers."""
    program = _fold_constants_and_cse(program)
    program = _eliminate_dead_code(program)
    program = _allocate_registers(program)
    return program

def precompile(roots: Sequence[Node], bindings: Mapping = {}) -> Program:
    """Translate expression graphs into an unoptimized program."""
    variables = {}
    runtime_symbols = {}
    instructions = []
    compiled: dict[int, Operand] = {} # cache for each compiled node

    for node in walk_ast(roots, bindings):
        if isinstance(node, Parameter) and node.is_variable:
            if id(node) not in variables:
                variables[id(node)] = len(variables)

        elif isinstance(node, Symbol) and node.key not in bindings:
            if node.key not in runtime_symbols:
                runtime_symbols[node.key] = len(runtime_symbols)

    def compile_node(node: Node) -> Operand:
        node_id = id(node)

        if node_id in compiled:
            return compiled[node_id]

        if isinstance(node, Literal):
            operand = Constant(node.value)

        elif isinstance(node, Parameter):
            operand = Input(variables[node_id]) if node.is_variable else Constant(node.value)

        elif isinstance(node, Symbol):
            if node.key in bindings:
                value = bindings[node.key]
                operand = compile_node(value) if isinstance(value, Node) else Constant(value)
            else:
                index = len(variables) + runtime_symbols[node.key]
                operand = Input(index)

        elif isinstance(node, InputNode):
            if node.node is None:
                raise UnresolvedInputError("Encountered an empty InputNode.")
            operand = compile_node(node.node)

        elif isinstance(node, UnaryOp):
            src = compile_node(node.operand)
            dst = len(instructions)
            instructions.append(UnaryInstruction(node.op, dst, src))
            operand = Register(dst)

        elif isinstance(node, BinaryOp):
            left = compile_node(node.left)
            right = compile_node(node.right)
            dst = len(instructions)
            instructions.append(BinaryInstruction(node.op, dst, left, right))
            operand = Register(dst)

        else:
            raise UnsupportedNodeError(f"Unsupported AST node type: {type(node).__name__}")

        compiled[node_id] = operand
        return operand
    
    outputs = tuple(compile_node(root) for root in roots)

    return Program(
        variables=variables,
        runtime_symbols=runtime_symbols,
        _instructions=tuple(instructions),
        _outputs=outputs,
        _register_count=len(instructions),
    )
    

def fuse(*programs: Program) -> Program:
    """Concatenate multiple independent programs into one program.

    Assumes all programs use the same input convention for theta and runtime
    symbol bindings.

    The fused program returns the concatenated outputs of all input programs,
    in the same order the programs were passed in.
    """
    if not programs:
        raise ValueError("Expected at least one program.")

    # TODO all programs must expect the same exact symbol and parameter mapping. Otherwise we cant fuse

    variables = {}
    runtime_symbols = {}
    instructions = []
    outputs = []

    register_offset = 0

    def remap_operand(operand: Operand, offset: int) -> Operand:
        if isinstance(operand, Register):
            return Register(operand.index + offset)
        if isinstance(operand, Input):
            return operand
        if isinstance(operand, Constant):
            return operand
        raise TypeError(f"Unknown operand: {type(operand).__name__}")

    for program in programs:
        variables.update(program.variables)
        runtime_symbols.update(program.runtime_symbols)

        for instruction in program._instructions:
            if isinstance(instruction, UnaryInstruction):
                instructions.append(
                    UnaryInstruction(
                        op=instruction.op,
                        dst=instruction.dst + register_offset,
                        src=remap_operand(instruction.src, register_offset),
                    )
                )
            elif isinstance(instruction, BinaryInstruction):
                instructions.append(
                    BinaryInstruction(
                        op=instruction.op,
                        dst=instruction.dst + register_offset,
                        left=remap_operand(instruction.left, register_offset),
                        right=remap_operand(instruction.right, register_offset),
                    )
                )
            else:
                raise TypeError(f"Unknown instruction: {type(instruction).__name__}")

        outputs.extend(remap_operand(operand, register_offset) for operand in program._outputs)
        register_offset += program._register_count

    return Program(
        variables=variables,
        runtime_symbols=runtime_symbols,
        _instructions=tuple(instructions),
        _outputs=tuple(outputs),
        _register_count=register_offset,
    )


#############################
# OPTIMIZATION IMPLEMENTATION
#############################

def _replace_operand(operand, replacements):
    if isinstance(operand, Register):
        return replacements[operand.index]
    return operand

def _sources(instruction):
    """Return instruction operands in evaluation order."""
    if isinstance(instruction, UnaryInstruction):
        return (instruction.src,)
    if isinstance(instruction, BinaryInstruction):
        return instruction.left, instruction.right
    raise TypeError(f"Unknown instruction: {type(instruction).__name__}")

def _operand_key(operand: Operand):
    """Return a key for CSE, or None if the operand cannot be compared safely."""
    if isinstance(operand, (Input, Register)):
        return type(operand), operand.index

    value = operand.value

    if type(value) not in (bool, int, float, complex):
        return None
    if isinstance(value, (float, complex)) and not np.isfinite(value):
        return None

    return Constant, type(value), repr(value)

def _rewrite_instruction(instruction, destination, operands):
    if isinstance(instruction, UnaryInstruction):
        return UnaryInstruction(instruction.op, destination, operands[0])
    return BinaryInstruction(instruction.op, destination, operands[0], operands[1])

def _fold_constants_and_cse(program: Program) -> Program:
    """Fold constant expressions and merge equivalent instructions."""
    instructions = []
    replacements = {}
    computations = {}

    for instruction in program._instructions:
        operands = tuple(
            _replace_operand(operand, replacements) for operand in _sources(instruction)
        )
        keys = tuple(_operand_key(operand) for operand in operands)
        comparable = all(key is not None for key in keys)

        # fold constants
        if comparable and all(isinstance(operand, Constant) for operand in operands):
            try:
                value = instruction.op.func(*(operand.value for operand in operands))
                constant = Constant(value)

                if _operand_key(constant) is not None:
                    replacements[instruction.dst] = constant
                    continue

            except (ArithmeticError, TypeError, ValueError):
                # Leave operations that cannot be folded for runtime evaluation.
                pass
        
        # common subexpression elimination
        key = (type(instruction), instruction.op, keys)
        if comparable and key in computations:
            replacements[instruction.dst] = computations[key]
            continue

        destination = len(instructions)
        instructions.append(_rewrite_instruction(instruction, destination, operands))
        replacement = Register(destination)
        replacements[instruction.dst] = replacement
        if comparable:
            computations[key] = replacement

    outputs = tuple(_replace_operand(output, replacements) for output in program._outputs)
    return Program(
        variables=dict(program.variables),
        runtime_symbols=dict(program.runtime_symbols),
        _instructions=tuple(instructions),
        _outputs=outputs,
        _register_count=len(instructions),
    )


def _eliminate_dead_code(program: Program) -> Program:
    """Remove instructions that do not contribute to an output."""
    live = set()
    for output in program._outputs:
        if isinstance(output, Register):
            live.add(output.index)

    retained = []
    for instruction in reversed(program._instructions):
        if instruction.dst not in live:
            continue
        retained.append(instruction)
        for operand in _sources(instruction):
            if isinstance(operand, Register):
                live.add(operand.index)

    instructions = []
    replacements = {}
    for instruction in reversed(retained):
        operands = tuple(
            _replace_operand(operand, replacements)
            for operand in _sources(instruction)
        )
        destination = len(instructions)
        instructions.append(_rewrite_instruction(instruction, destination, operands))
        replacements[instruction.dst] = Register(destination)

    outputs = tuple(_replace_operand(output, replacements) for output in program._outputs)
    return Program(
        variables=dict(program.variables),
        runtime_symbols=dict(program.runtime_symbols),
        _instructions=tuple(instructions),
        _outputs=outputs,
        _register_count=len(instructions),
    )


def _allocate_registers(program: Program) -> Program:
    """Reuse registers after their final use."""
    last_use = {}
    for index, instruction in enumerate(program._instructions):
        for operand in _sources(instruction):
            if isinstance(operand, Register):
                last_use[operand.index] = index

    for output in program._outputs:
        if isinstance(output, Register):
            last_use[output.index] = len(program._instructions)

    instructions = []
    replacements = {}
    available = []
    register_count = 0

    for index, instruction in enumerate(program._instructions):
        sources = _sources(instruction)
        operands = tuple(_replace_operand(source, replacements) for source in sources)

        released = set()
        for source in sources:
            if not isinstance(source, Register) or source.index in released:
                continue
            if last_use[source.index] == index:
                available.append(replacements[source.index].index)
                released.add(source.index)

        # Evaluation reads all sources before writing the destination.
        if available:
            destination = available.pop()
        else:
            destination = register_count
            register_count += 1

        instructions.append(_rewrite_instruction(instruction, destination, operands))
        replacements[instruction.dst] = Register(destination)
        if instruction.dst not in last_use:
            available.append(destination)

    outputs = tuple(_replace_operand(output, replacements) for output in program._outputs)
    return Program(
        variables=dict(program.variables),
        runtime_symbols=dict(program.runtime_symbols),
        _instructions=tuple(instructions),
        _outputs=outputs,
        _register_count=register_count,
    )
