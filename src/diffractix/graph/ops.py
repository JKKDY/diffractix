from enum import Enum, auto
import autograd.numpy as np

class Op(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    MAX = auto()
    MIN = auto()
    SIGMOID = auto() # For soft constraints
    
    def __repr__(self):
        return self.name

    @property
    def is_commutative(self):
        return self in (Op.ADD, Op.MUL)

    @property
    def unicode(self) -> str:
        return {
            Op.ADD: "＋",
            Op.SUB: "−",
            Op.MUL: "×",
            Op.DIV: "÷",
            Op.NEG: "−",
            Op.MAX: "max",
            Op.MIN: "min",
            Op.SIGMOID: "σ",
        }[self]

OP_MAP = {
    Op.ADD: np.add,
    Op.SUB: np.subtract,
    Op.MUL: np.multiply,
    Op.DIV: np.divide,
    Op.NEG: np.negative,
    Op.MAX: np.maximum,
    Op.MIN: np.minimum,
}

