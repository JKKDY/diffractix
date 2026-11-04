from collections.abc import Mapping, Sequence


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
from dataclasses import replace


def remap_parameters(program: Program, parameter_indices: Mapping[int, int]) -> Program:
    """Return a program that reads parameters from the given theta indices."""
    parameter_indices = dict(parameter_indices)
    if set(parameter_indices.values()) != set(range(len(parameter_indices))):
        raise ValueError("Parameter indices must be consecutive starting at zero.")

    old_size = len(program.parameter_indices)
    new_size = len(parameter_indices)

    old_indices_inverted = {
        index: parameter_id
        for parameter_id, index in program.parameter_indices.items()
    }

    def remap(operand):
        if not isinstance(operand, Input):
            return operand
        if operand.index >= old_size:
            return Input(new_size - old_size + operand.index)
        if parameter_id:=old_indices_inverted[operand.index] not in parameter_indices:
            raise ValueError(f"Missing required Parameter with id {parameter_id}.")
        return Input(parameter_indices[parameter_id])

    instructions = []
    for instruction in program._instructions:
        if isinstance(instruction, UnaryInstruction):
            instructions.append(replace(
                instruction, 
                src=remap(instruction.src)
            ))
        elif isinstance(instruction, BinaryInstruction):
            instructions.append(replace(
                instruction,
                left=remap(instruction.left),
                right=remap(instruction.right),
            ))

    return replace(
        program,
        parameter_indices=parameter_indices,
        _instructions=tuple(instructions),
        _outputs=tuple(remap(output) for output in program._outputs)
    )


def remap_runtime_symbols(program: Program, runtime_symbols: Mapping[Hashable, int]) -> Program:
    """Return a program using the given runtime symbol indices."""
    runtime_symbols = dict(runtime_symbols)

    if not program.runtime_symbols.keys() <= runtime_symbols.keys():
        raise ValueError("Missing required runtime symbols.")

    input_size = len(program.parameter_indices)
    old_indices_inverted = {
        index: key
        for key, index in program.runtime_symbols.items()
    }

    def remap(operand):
        if not isinstance(operand, Input) or operand.index < input_size:
            return operand
        key = old_indices_inverted[operand.index - input_size]
        return Input(input_size + runtime_symbols[key])

    instructions = []
    for instruction in program._instructions:
        if isinstance(instruction, UnaryInstruction):
            instructions.append(replace(
                instruction,
                src=remap(instruction.src),
            ))
        elif isinstance(instruction, BinaryInstruction):
            instructions.append(replace(
                instruction,
                left=remap(instruction.left),
                right=remap(instruction.right),
            ))
        else:
            raise TypeError(f"Unknown instruction: {type(instruction).__name__}")

    return replace(
        program,
        runtime_symbols=runtime_symbols,
        _instructions=tuple(instructions),
        _outputs=tuple(remap(output) for output in program._outputs)
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

    for program in programs[1:]:
        if program.parameter_indices != programs[0].parameter_indices:
            raise ValueError(
                "Programs must have identical parameter mappings to fuse."
                f"Got {program.parameter_indices} and {programs[0].parameter_indices}"
            )
        if program.runtime_symbols != programs[0].runtime_symbols:
            raise ValueError(
                f"Programs must have identical symbol mappings to fuse. "
                f"Got {program.runtime_symbols} and {programs[0].runtime_symbols}"
            )

    parameter_indices = {}
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
        parameter_indices.update(program.parameter_indices)
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
        parameter_indices=parameter_indices,
        runtime_symbols=runtime_symbols,
        _instructions=tuple(instructions),
        _outputs=tuple(outputs),
        _register_count=register_offset,
    )