from __future__ import annotations
from abc import abstractmethod
import weakref
from .ops import Op
import weakref


Scalar = int | float


class Node:
    """
    Base class for all nodes in the Abstract Syntax Tree.
    Handles operator overloading to construct the graph dynamically.
    """

    def __add__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.ADD, self, other)

    def __sub__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.SUB, self, other)

    def __mul__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.MUL, self, other)

    def __truediv__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.DIV, self, other)

    def __neg__(self) -> UnaryOp:
        return UnaryOp(Op.NEG, self)


    def __radd__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.ADD, other, self)

    def __rmul__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.MUL, other, self)

    def __rsub__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.SUB, other, self)

    def __rtruediv__(self, other: Node | Scalar) -> BinaryOp:
        return BinaryOp(Op.DIV, other, self)


    def maximum(self, floor: float) -> BinaryOp: 
        return BinaryOp(Op.MAX, self, floor)

    def minimum(self, ceiling: float)-> BinaryOp: 
        return BinaryOp(Op.MIN, self, ceiling)



class BinaryOp(Node):
    """
    Represents an operation taking two operands (e.g., a + b).
    """
    def __init__(self, op: Op, left: Node | Scalar, right: Node | Scalar):
        self.op = op
        self.left = left 
        self.right = right 

    def __repr__(self) -> str:
        return f"({self.left} {self.op.name} {self.right})"


class UnaryOp(Node):
    """
    Represents an operation taking a single operand (e.g., -a).
    """
    def __init__(self, op: Op, operand: Node | Scalar):
        self.op = op
        self.operand = operand 
    
    def __repr__(self): 
        return f"{self.op.name}({self.operand})"



class Parameter(Node):
    """
    Represents a physical property (d, f, n) of an optical element.
    It is a Node in the graph that holds a value.
    
    State:
        - value: The current number (float).
        - fixed: If False, this parameter is added to the Optimizer Vector else if True treat it as a constant
    """
    def __init__(self, value: float, name: str, fixed: bool, owner: "OpticalElement"):
        self.value = float(value)
        self.name = name
        self.fixed = fixed
        # need a reference to the owner (a optical element) of this parameter to correctly get the name
        # any non referential methods introduce problems with dataclass and the initialization order of members in OpticalElement
        self._owner_ref = weakref.ref(owner) # weakref to avoid any circular dependencies

    @property
    def owner(self):
        return self._owner_ref()

    @property
    def full_name(self):
        """Reconstructs the full name dynamically."""
        if hasattr(self.owner, 'label') and self.owner.label:
            return f"{self.owner.label}.{self.name}"
        return self.name

    def __repr__(self):
        tag = "[FIX]" if self.fixed else "[VAR]"
        return f"{self.full_name} {tag}={self.value:.4g}"
