from __future__ import annotations
from abc import abstractmethod
import weakref
from .ops import Op
import weakref
import itertools


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
      

    # UNARY
    def __neg__(self) -> UnaryOp:
        return Node._make_unary_op(Op.NEG, self)

    def __pos__(self) -> UnaryOp:
        return Node._make_unary_op(Op.POS, self)

    def __abs__(self) -> UnaryOp:
        return Node._make_unary_op(Op.ABS, self)

    def sigmoid(self) -> UnaryOp:
        return Node._make_unary_op(Op.SIGMOID, self)


    # ADDITION
    def __add__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, self, other)

    def __radd__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, other, self)


    # SUBTRACTION
    def __sub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, self, other)

    def __rsub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, other, self)


    # MULTIPLICATION
    def __mul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, self, other)

    def __rmul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, other, self)


    # TRUE DIVISION
    def __truediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, self, other)

    def __rtruediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, other, self)


    # FLOOR DIVISION 
    def __floordiv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.FLOORDIV, self, other)

    def __rfloordiv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.FLOORDIV, other, self)


    # MODULO
    def __mod__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MOD, self, other)

    def __rmod__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MOD, other, self)


    # POWER
    def __pow__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.POW, self, other)

    def __rpow__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.POW, other, self)


    # EXTREMA
    def maximum(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MAX, self, other)

    def minimum(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MIN, self, other)

    def __float__(self):
        if not hasattr(self, "value"):
            raise Exception(f"self.value not implemented for {self.__class__}")
        return float(self.value)





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

    @property
    def value(self) -> float:
        return self.op.func(self.left.value, self.right.value) 

    @property
    def is_constant(self):
        return self.left.is_constant and self.right.is_constant

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, BinaryOp): return False
        
        # operators must match
        if self.op != other.op:
            return False
            
        # check operands
        if self.op.is_commutative:
            # for commutative ops, we check "cross" equality
            return (
                (self.left == other.left and self.right == other.right) or
                (self.left == other.right and self.right == other.left)
            )
        else:
            # Strict order for non-commutative
            return self.left == other.left and self.right == other.right


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

    @property
    def value(self) -> float:
        return self.op.func(self.operand.value) 

    @property
    def is_constant(self):
        return self.operand.is_constant

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, UnaryOp): return False
        return self.op == other.op and self.operand == other.operand




class InputNode(Node):
    def __init__(self, node: Paramter | Constant | Symbol): 
        self.node = node.node if isinstance(node, InputNode) else node

    def __getattr__(self, name):
        return getattr(self.node, name)

    def __setattr__(self, name, value):
        # If we are changing the 'body' of the handle, do it normally
        if name == "node":
            super().__setattr__('node', value)
        else:
            # Redirect all other writes (e.g., .fixed, .value, .name) to the inner node
            setattr(self.node, name, value)

    def __repr__(self): 
        return f"Input:{self.node}"

    @property
    def is_constant(self):
        return self.node.is_constant

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.node)



class Constant(Node):
    def __init__(self, value: float):
        self.value = float(value)

    def __hash__(self):
        print("sjdkkjsdfn")
        print(hash((Constant, hash(self.value))))
        return hash((Constant, hash(self.value)))

    def __repr__(self):
        return f"Const={self.value:.4g}"

    @property
    def is_constant(self):
        return True

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, Constant): return False
        return self.value == other.value


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


    def __init__(self, value: float, name: str, fixed: bool, owner: "OpticalElement" = None):
        if (value is None):
            pass

        self.id = next(Parameter._id_counter)
        self.value = float(value)
        self.name = name
        self.fixed = fixed
        # need a reference to the owner (a optical element) of this parameter to correctly get the name
        # any non referential methods introduce problems with dataclass and the initialization order of members in OpticalElement
        if owner: self._owner_ref = weakref.ref(owner) # weakref to avoid any circular dependencies

    @property
    def owner(self):
        if not hasattr(self, "_owner_ref"):
            raise ValueError(f"This paremeter {self.name} has no owner")
        return self._owner_ref()

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return self is other

    @property
    def full_name(self):
        """Reconstructs the full name dynamically."""
        return f"{self.owner.label}.{self.name}"

    def __repr__(self):
        return f"{self.full_name}{'(F)' if self.fixed else '(V)'}={self.value:.4g}"

    @property
    def is_constant(self):
        return self.fixed

    def __eq__(self, other):
        return self is other


class Symbol(Node):
    def __new__(cls, name: str):
        # we check if this symbold is already in the class
        # the logic here differs from the standard Node._register because we must ensurure that we dont
        # override any bindings (i.e. set self.taget to None)
        
        # cehck cachse
        key = hash((Symbol, name))
        if key in Node._cache:
            return Node._cache[key]
        
        # create New if missing
        instance = super(Node, cls).__new__(cls)
        instance.name = name
        instance.target = None # The future input
        
        # and register explicitly
        Node._cache[key] = instance
        return instance

    def __init__(self, name:str):
        self.name = name
        self.fixed = None
        self.target = None


    @property
    def value(self):
        if self.target is None:
            raise ValueError(f"Symbol '{self.name}' has not been bound to a value yet.")
        return self.target.value

    @value.setter
    def value(self, v):
        if self.target is None:
            raise ValueError(f"Symbol '{self.name}' has not been bound to a value yet.")
        self.target.value = v

    @property
    def is_constant(self):
        if self.target is None: return True # Default assumption until bound
        return self.target.is_constant
        
    def __repr__(self):
        if self.target:
            return f"Symbol({self.name} -> {self.target.value})"
        return f"Symbol({self.name} -> ?)"
    
    def __hash__(self):
        return hash((Symbol, self.name))

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, Symbol): return False
        return self.name == other.name

Scalar = int | float
ASTNode = Node
