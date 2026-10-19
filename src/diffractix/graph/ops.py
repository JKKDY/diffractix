from enum import Enum, auto
from functools import lru_cache
from collections.abc import Callable


from diffractix.backend import xp


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
    SINC = auto()

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
    def func(self):
        return get_op_functions()[self]


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
    Op.SINC: 1,
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
    Op.SINC: "sinc",
    Op.MAX: "max",
    Op.MIN: "min",
}


@lru_cache(maxsize=1)
def get_op_functions():
    return {
        Op.ADD: xp.add,
        Op.SUB: xp.subtract,
        Op.MUL: xp.multiply,
        Op.DIV: xp.divide,
        Op.POW: xp.power,
        Op.NEG: xp.negative,
        Op.POS: lambda x: x,
        Op.ABS: xp.abs,
        Op.SIGMOID: lambda x: 1.0 / (1.0 + xp.exp(-x)),
        Op.EXP: xp.exp,
        Op.LOG: xp.log,
        Op.SQRT: xp.sqrt,
        Op.SIN: xp.sin,
        Op.COS: xp.cos,
        Op.TAN: xp.tan,
        Op.SINH: xp.sinh,
        Op.COSH: xp.cosh,
        Op.TANH: xp.tanh,
        Op.SINC: lambda x: xp.where(
            x == 0,
            1.0,
            xp.sin(x) / xp.where(x == 0, 1.0, x),
        ),
        Op.MAX: xp.maximum,
        Op.MIN: xp.minimum,
    }