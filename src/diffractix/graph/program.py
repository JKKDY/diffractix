from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Hashable
from dataclasses import dataclass
from enum import Enum, auto

from typing import Any

import numpy as np
from diffractix.backend import xp


from .node import (
    Node,
    Literal,
    Parameter,
    Symbol,
    InputNode,
    BinaryOp,
    UnaryOp,
    Scalar
)

from .ops import Op



@dataclass(frozen=True)
class Input:
    index: int

@dataclass(frozen=True)
class Register:
    index: int

@dataclass(frozen=True)
class Constant:
    value: int | float | complex

Operand = Input | Register | Constant

@dataclass(frozen=True)
class UnaryInstruction:
    op: Op
    dst: int
    src: Operand


@dataclass(frozen=True)
class BinaryInstruction:
    op: Op
    dst: int
    left: Operand
    right: Operand

Instruction = UnaryInstruction | BinaryInstruction


@dataclass(frozen=True)
class Program:
    """
    Compiled numerical program for evaluating expression graphs.

    Inputs contain numerical variables followed by runtime symbols.
    Instructions operate on these inputs and produce an output vector.
    """
    variables: dict[int, int]       # id(Parameter) -> theta index
    runtime_symbols: dict[Hashable, int]     # symbol key -> symbol index

    _instructions: tuple[Instruction, ...]
    _outputs: tuple[Operand, ...]
    _register_count: int



    def evaluate(self, theta: Sequence, bindings: Mapping = {}) -> tuple:
        """
        Execute the program with the given inputs.

        Parameters
        ----------
        theta:
            numerical variable values in the compiled parameter order.
        bindings:
            Mapping of runtime symbol keys to their values, for example
            {"wavelength": 1064e-9, "temperature": 293.15}.

        Returns
        -------
        tuple
            Computed outputs in their compiled order.
        """
        if len(theta) != len(self.variables):
            raise ValueError(
                f"Expected {len(self.variables)} variable values, got {len(theta)}."
            )
        
        sorted_symbols = sorted(self.runtime_symbols.items(), key=lambda item: item[1])
        sorted_bindings = (bindings[key] for key, _ in sorted_symbols)
        inputs = (*theta, *sorted_bindings)

        registers = [None] * self._register_count

        def read(operand: Operand) -> Any:
            if isinstance(operand, Input):
                return inputs[operand.index]
            if isinstance(operand, Register):
                return registers[operand.index]
            if isinstance(operand, Immediate):
                return operand.value
            raise TypeError(f"Unknown operand: {type(operand).__name__}")

        for instruction in self._instructions:
            if isinstance(instruction, UnaryInstruction):
                result = instruction.op.func(read(instruction.src))
            elif isinstance(instruction, BinaryInstruction):
                result = instruction.op.func(
                    read(instruction.left),
                    read(instruction.right),
                )
            else:
                raise TypeError(f"Unknown instruction: {type(instruction).__name__}")

            registers[instruction.dst] = result

        return tuple(read(operand) for operand in self._outputs)


    def __str__(self) -> str:
        def format_operand(operand):
            if isinstance(operand, Register):
                return f"r{operand.index}"
            if isinstance(operand, Input):
                return f"i{operand.index}"
            if isinstance(operand, Immediate):
                return repr(operand.value)
            raise TypeError(f"Unknown operand: {type(operand).__name__}")

        lines = ["INPUTS"]
        sorted_variables = sorted(self.variables.items(), key=lambda item: item[1])
        sorted_symbols = sorted(self.runtime_symbols.items(), key=lambda item: item[1])

        for parameter, index in sorted_variables:
            comment = f"\t# {parameter.name}" if parameter.name else ""
            lines.append(f"  i{index}\ttheta[{index}]{comment}")

        for key, index in sorted_symbols:
            lines.append(f"  i{len(self.variables) + index}\tsymbol[{key!r}]")

        lines.extend((
                f"\nREGISTERS  {self._register_count}",
                "\nPROGRAM",
            ))

        for instruction in self._instructions:
            if isinstance(instruction, UnaryInstruction):
                args = f"r{instruction.dst}, {format_operand(instruction.src)}"
            elif isinstance(instruction, BinaryInstruction):
                args = (
                    f"r{instruction.dst}, "
                    f"{format_operand(instruction.left)}, "
                    f"{format_operand(instruction.right)}"
                )
            else:
                raise TypeError(f"Unknown instruction: {type(instruction).__name__}")

            lines.append(f"  {instruction.op.name:<6} {args}")

        lines.append("\nOUTPUTS")
        for index, operand in enumerate(self._outputs):
            lines.append(f"  [{index}]\t{format_operand(operand)}")

        return "\n".join(lines)
