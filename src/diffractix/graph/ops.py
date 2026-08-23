from enum import Enum, auto

import autograd.numpy as np


class Op(Enum):
    # binary arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()

    # unary arithmetic
    NEG = auto()
    POS = auto()
    ABS = auto()

    # unary functions
    SIGMOID = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()

    # binary extrema
    MAX = auto()
    MIN = auto()

    def __repr__(self):
        return self.name

    @property
    def arity(self) -> int:
        return {
            Op.NEG: 1,
            Op.POS: 1,
            Op.ABS: 1,
            Op.SIGMOID: 1,
            Op.EXP: 1,
            Op.LOG: 1,
            Op.SQRT: 1,
        }.get(self, 2)

    @property
    def is_commutative(self) -> bool:
        return self in {
            Op.ADD,
            Op.MUL,
            Op.MAX,
            Op.MIN,
        }

    @property
    def unicode(self) -> str:
        return {
            Op.ADD: "＋",
            Op.SUB: "−",
            Op.MUL: "×",
            Op.DIV: "÷",
            Op.FLOORDIV: "⌊÷⌋",
            Op.MOD: "mod",
            Op.POW: "^",
            Op.NEG: "−",
            Op.POS: "+",
            Op.ABS: "|·|",
            Op.SIGMOID: "σ",
            Op.EXP: "exp",
            Op.LOG: "log",
            Op.SQRT: "√",
            Op.MAX: "max",
            Op.MIN: "min",
        }[self]

    @property
    def func(self) -> callable:
        return {
            Op.ADD: np.add,
            Op.SUB: np.subtract,
            Op.MUL: np.multiply,
            Op.DIV: np.divide,
            Op.FLOORDIV: np.floor_divide,
            Op.MOD: np.mod,
            Op.POW: np.power,
            Op.NEG: np.negative,
            Op.POS: lambda x: x,
            Op.ABS: np.abs,
            Op.SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-x)),
            Op.EXP: np.exp,
            Op.LOG: np.log,
            Op.SQRT: np.sqrt,
            Op.MAX: np.maximum,
            Op.MIN: np.minimum,
        }[self]