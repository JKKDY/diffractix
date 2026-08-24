from __future__ import annotations

import weakref
import autograd.numpy as np

from .ops import Op


Scalar = int | float | complex


class Node:
    """
    Base class for all nodes in the parameter AST.

    Nodes use normal Python object identity. Structurally equivalent
    expressions are not deduplicated during graph construction.
    """

    @staticmethod
    def _make_literal(value: Scalar) -> Literal:
        if not isinstance(value, (int, float, complex)):
            raise TypeError(f"Expected numeric scalar, got {type(value)}")
        return Literal(value)

    @staticmethod
    def _make_binary_op(
        op: Op,
        left: Node | Scalar,
        right: Node | Scalar,
    ) -> BinaryOp:
        if not isinstance(left, Node):
            left = Node._make_literal(left)

        if not isinstance(right, Node):
            right = Node._make_literal(right)

        return BinaryOp(op, left, right)

    @staticmethod
    def _make_unary_op(
        op: Op,
        operand: Node | Scalar,
    ) -> UnaryOp:
        if not isinstance(operand, Node):
            operand = Node._make_literal(operand)

        return UnaryOp(op, operand)

    # ----------------
    # Unary operations
    # ----------------
    def __neg__(self) -> UnaryOp:
        return Node._make_unary_op(Op.NEG, self)

    def __pos__(self) -> UnaryOp:
        return Node._make_unary_op(Op.POS, self)

    def __abs__(self) -> UnaryOp:
        return Node._make_unary_op(Op.ABS, self)

    def sigmoid(self) -> UnaryOp:
        return Node._make_unary_op(Op.SIGMOID, self)

    def exp(self) -> UnaryOp:
        return Node._make_unary_op(Op.EXP, self)

    def log(self) -> UnaryOp:
        return Node._make_unary_op(Op.LOG, self)

    def sqrt(self) -> UnaryOp:
        return Node._make_unary_op(Op.SQRT, self)

    # --------
    # Addition
    # --------
    def __add__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, self, other)

    def __radd__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, other, self)

    # -----------
    # Subtraction
    # -----------
    def __sub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, self, other)

    def __rsub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, other, self)

    # --------------
    # Multiplication
    # --------------
    def __mul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, self, other)

    def __rmul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, other, self)

    # -------------
    # True division
    # -------------
    def __truediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, self, other)

    def __rtruediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, other, self)

    # --------------
    # Floor division
    # --------------
    def __floordiv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.FLOORDIV, self, other)

    def __rfloordiv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.FLOORDIV, other, self)

    # ------
    # Modulo
    # ------
    def __mod__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MOD, self, other)

    def __rmod__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MOD, other, self)

    # -----
    # Power
    # -----
    def __pow__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.POW, self, other)

    def __rpow__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.POW, other, self)

    # -------
    # Extrema
    # -------
    def maximum(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MAX, self, other)

    def minimum(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MIN, self, other)

    # ----------
    # Conversion
    # ----------
    def __float__(self):
        if not hasattr(self, "value"):
            raise TypeError(f"{self.__class__.__name__} does not provide a value")
        return float(self.value)


class BinaryOp(Node):
    """Expression containing a binary operation such as a + b."""

    def __init__(self, op: Op, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op.unicode} {self.right})"

    @property
    def value(self):
        return self.op.func(self.left.value, self.right.value)

    @property
    def is_variable(self):
        return self.left.is_variable or self.right.is_variable


class UnaryOp(Node):
    """Expression containing a unary operation such as -a."""

    def __init__(self, op: Op, operand: Node):
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"{self.op.unicode}({self.operand})"

    @property
    def value(self):
        return self.op.func(self.operand.value)

    @property
    def is_variable(self):
        return self.operand.is_variable


class InputNode(Node):
    """
    Stable handle to another AST node.

    The underlying node may be replaced without invalidating expressions
    that already reference this InputNode.
    """

    def __init__(self, node: Node | None):
        self.node = node

    def __getattr__(self, name):
        if self.node is None:
            raise AttributeError(
                f"InputNode is empty and has no attribute '{name}'"
            )
        return getattr(self.node, name)

    def __setattr__(self, name, value):
        if name == "node":
            super().__setattr__("node", value)
            return

        if self.node is None:
            raise AttributeError(
                f"Cannot set '{name}' on an empty InputNode"
            )

        setattr(self.node, name, value)

    def __repr__(self):
        return f"Input:{self.node}"

    @property
    def is_variable(self):
        return self.node.is_variable if self.node is not None else False


class Literal(Node):
    """
    Immutable numeric literal embedded in an expression.

    Example:
        y = 2 * x

    The value 2 is represented by a Literal rather than a Parameter.
    """

    def __init__(self, value: Scalar):
        if not isinstance(value, (int, float, complex)):
            raise TypeError("Literal value must be numeric.")

        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def is_variable(self):
        return False

    def __repr__(self):
        return repr(self._value)


class Parameter(Node):
    """
    Mutable scalar design value.

    Parameters may exist independently or originate from an OpticalElement.
    A variable Parameter becomes a degree of freedom during compilation.
    A fixed Parameter remains mutable but is excluded from the optimizer.
    """

    def __init__(
        self,
        value: float,
        name: str | None = None,
        *,
        variable: bool = False,
        min_val: float = -np.inf,
        max_val: float = np.inf,
        owner: "OpticalElement | None" = None,
    ):
        if min_val > max_val:
            raise ValueError("min_val must be <= max_val.")

        if value < min_val or value > max_val:
            raise ValueError(
                f"Parameter value {value} lies outside bounds "
                f"[{min_val}, {max_val}]."
            )

        self.value = float(value)
        self.name = name
        self._variable = variable
        self.min_val = min_val
        self.max_val = max_val
        self._owner_ref = weakref.ref(owner) if owner is not None else None

    @property
    def is_variable(self) -> bool:
        return self._variable

    def variable(self):
        self._variable = True
        return self

    def fixed(self):
        self._variable = False
        return self

    def bound(
        self,
        min_val: float = -np.inf,
        max_val: float = np.inf,
    ):
        if min_val > max_val:
            raise ValueError("min_val must be <= max_val.")

        if self.value < min_val or self.value > max_val:
            raise ValueError(
                f"Current value {self.value} lies outside bounds "
                f"[{min_val}, {max_val}]."
            )

        self.min_val = min_val
        self.max_val = max_val
        return self

    @property
    def owner(self):
        if self._owner_ref is None:
            return None
        return self._owner_ref()

    @property
    def full_name(self):
        owner = self.owner

        if owner is not None:
            if self.name is not None:
                return f"{owner.label}.{self.name}"
            return owner.label

        return self.name or "<parameter>"

    def __repr__(self):
        status = "V" if self.is_variable else "F"
        return f"{self.full_name}={self.value:.4g}[{status}]"


class SystemVar(Node):
    """
    Immutable reference to a value supplied by the System compilation context.

    SystemVars are not optimization variables and do not contain mutable
    bindings. The compiler resolves them against the current System.
    """

    def __init__(self, name: str, *, namespace: str = "system"):
        if not isinstance(name, str) or not name:
            raise ValueError("SystemVar name must be a non-empty string.")

        self._name = name
        self._namespace = namespace

    @property
    def name(self):
        return self._name

    @property
    def namespace(self):
        return self._namespace

    @property
    def is_variable(self):
        return False

    @property
    def value(self):
        raise RuntimeError(
            f"{self!r} has no standalone value. "
            "SystemVars are resolved by System.build()."
        )

    def __repr__(self):
        if self.namespace == "system":
            return f"SystemVar({self.name!r})"
        return f"SystemVar({self.namespace!r}, {self.name!r})"


ASTNode = Node


