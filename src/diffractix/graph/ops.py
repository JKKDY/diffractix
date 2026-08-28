from enum import Enum, auto
from collections.abc import Callable

import autograd.numpy as np


class Op(Enum):
    # binary arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
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
    SIN = auto()
    COS = auto()
    TAN = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()

    # binary extrema
    MAX = auto()
    MIN = auto()

    def __repr__(self):
        return self.name

    @property
    def arity(self) -> int:
        return OP_ARITY[self]

    @property
    def is_commutative(self) -> bool:
        return self in COMMUTATIVE_OPS

    @property
    def unicode(self) -> str:
        return OP_UNICODE[self]

    @property
    def func(self) -> Callable:
        return OP_FUNCTIONS[self]


OP_ARITY = {
    Op.ADD: 2,
    Op.SUB: 2,
    Op.MUL: 2,
    Op.DIV: 2,
    Op.POW: 2,
    Op.NEG: 1,
    Op.POS: 1,
    Op.ABS: 1,
    Op.SIGMOID: 1,
    Op.EXP: 1,
    Op.LOG: 1,
    Op.SQRT: 1,
    Op.SIN: 1,
    Op.COS: 1,
    Op.TAN: 1,
    Op.SINH: 1,
    Op.COSH: 1,
    Op.TANH: 1,
    Op.MAX: 2,
    Op.MIN: 2,
}


COMMUTATIVE_OPS = {
    Op.ADD,
    Op.MUL,
    Op.MAX,
    Op.MIN,
}


OP_UNICODE = {
    Op.ADD: "＋",
    Op.SUB: "−",
    Op.MUL: "×",
    Op.DIV: "÷",
    Op.POW: "^",
    Op.NEG: "−",
    Op.POS: "+",
    Op.ABS: "|·|",
    Op.SIGMOID: "σ",
    Op.EXP: "exp",
    Op.LOG: "log",
    Op.SQRT: "√",
    Op.SIN: "sin",
    Op.COS: "cos",
    Op.TAN: "tan",
    Op.SINH: "sinh",
    Op.COSH: "cosh",
    Op.TANH: "tanh",
    Op.MAX: "max",
    Op.MIN: "min",
}


OP_FUNCTIONS = {
    Op.ADD: np.add,
    Op.SUB: np.subtract,
    Op.MUL: np.multiply,
    Op.DIV: np.divide,
    Op.POW: np.power,
    Op.NEG: np.negative,
    Op.POS: lambda x: x,
    Op.ABS: np.abs,
    Op.SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-x)),
    Op.EXP: np.exp,
    Op.LOG: np.log,
    Op.SQRT: np.sqrt,
    Op.SIN: np.sin,
    Op.COS: np.cos,
    Op.TAN: np.tan,
    Op.SINH: np.sinh,
    Op.COSH: np.cosh,
    Op.TANH: np.tanh,
    Op.MAX: np.maximum,
    Op.MIN: np.minimum,
}