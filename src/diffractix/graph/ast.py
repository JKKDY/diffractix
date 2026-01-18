from __future__ import annotations
from abc import abstractmethod
import weakref
from .ops import Op
import weakref
import itertools

Scalar = int | float


class Node:
    """
    Base class for all nodes in the Abstract Syntax Tree.
    Handles operator overloading to construct the graph dynamically.
    """
    _cache = weakref.WeakValueDictionary()


    @staticmethod 
    def _register(obj):
        key = hash(obj)

        if key in Node._cache:
            return Node._cache[key]
            
        Node._cache[key] = obj
        return obj

    @staticmethod
    def _make_constant(value: Scalar) -> Constant:
        return Node._register(Constant(value))

    @staticmethod
    def _make_binary_op(op: Op, left: Node | Scalar, right: Node | Scalar):
        if not isinstance(left, Node): left = Node._make_constant(left)
        if not isinstance(right, Node): right = Node._make_constant(right)

        return Node._register(BinaryOp(op, left, right))

    @staticmethod
    def _make_unary_op(op: Op, operand: Node | Scalar):
        if not isinstance(operand, Node): operand = Node._make_constant(operand)
        return Node._register(UnaryOp(op, operand))
      

    def __add__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, self, other)

    def __sub__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.SUB, self, other)

    def __mul__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.MUL, self, other)

    def __truediv__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.DIV, self, other)

    def __neg__(self) -> UnaryOp:
        return  Node._make_unary_op(Op.NEG, self)


    def __radd__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.ADD, other, self)

    def __rmul__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.MUL, other, self)

    def __rsub__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.SUB, other, self)

    def __rtruediv__(self, other: Node | Scalar) -> BinaryOp:
        return  Node._make_binary_op(Op.DIV, other, self)


    def maximum(self, floor: float) -> BinaryOp: 
        return  Node._make_binary_op(Op.MAX, self, floor)

    def minimum(self, ceiling: float)-> BinaryOp: 
        return  Node._make_binary_op(Op.MIN, self, ceiling)



class BinaryOp(Node):
    """
    Represents an operation taking two operands (e.g., a + b).
    """
    def __init__(self, op: Op, left: Node | Scalar, right: Node | Scalar):
        self.op = op
        self.left = left 
        self.right = right 

    def __repr__(self) -> str:
        return f"({self.left} {self.op.unicode} {self.right})"

    def __hash__(self):
        l, r = hash(self.left), hash(self.right)
        if self.op.is_commutative and l > r:
            l, r = r, l
        return hash((BinaryOp, self.op, l, r))

    def __eq__(self, other):
       return (isinstance(other, BinaryOp) and 
                self.op == other.op and 
                self.left is other.left and 
                self.right is other.right)


class UnaryOp(Node):
    """
    Represents an operation taking a single operand (e.g., -a).
    """
    def __init__(self, op: Op, operand: Node | Scalar):
        self.op = op
        self.operand = operand 
    
    def __repr__(self): 
        return f"{self.op.unicode}({self.operand})"

    def __hash__(self):
        return hash((UnaryOp, self.op, self.operand))

    def __eq__(self, other):
        return (isinstance(other, UnaryOp) and 
                self.op == other.op and 
                self.operand is other.operand)



class Constant(Node):
    def __init__(self, value: float):
        self.value = float(value)

    def __hash__(self):
        return hash((Constant, self.value))

    def __repr__(self):
        return f"{self.value:.4g}"


class Parameter(Node):
    """
    Represents a physical property (d, f, n) of an optical element.
    It is a Node in the graph that holds a value.
    
    State:
        - value: The current number (float).
        - fixed: If False, this parameter is added to the Optimizer Vector else if True treat it as a constant
    """
    # Global counter for unique IDs across the session
    _id_counter = itertools.count()

    def __init__(self, value: float, name: str, fixed: bool, owner: "OpticalElement"):
        self.id = next(Parameter._id_counter)
        self.value = float(value)
        self.name = name
        self.fixed = fixed
        # need a reference to the owner (a optical element) of this parameter to correctly get the name
        # any non referential methods introduce problems with dataclass and the initialization order of members in OpticalElement
        self._owner_ref = weakref.ref(owner) # weakref to avoid any circular dependencies

    @property
    def owner(self):
        return self._owner_ref()

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return self is other

    @property
    def full_name(self):
        """Reconstructs the full name dynamically."""
        if hasattr(self.owner, 'label') and self.owner.label:
            return f"{self.owner.label}.{self.name}"
        return self.name

    def __repr__(self):
        # tag = "[FIX]" if self.fixed else "[VAR]"
        return f"{self.full_name}={self.value:.4g}"
