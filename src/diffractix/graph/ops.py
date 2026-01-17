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

OP_MAP = {
    Op.ADD: np.add,
    Op.SUB: np.subtract,
    Op.MUL: np.multiply,
    Op.DIV: np.divide,
    Op.NEG: np.negative,
    Op.MAX: np.maximum,
    Op.MIN: np.minimum,
}

COMMUTATION_MAP = {
    Op.ADD: True,
    Op.SUB: False,
    Op.MUL: True,
    Op.DIV: False,
    Op.NEG: True,
    Op.MAX: True,
    Op.MIN: True,
}