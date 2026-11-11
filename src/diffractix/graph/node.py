from __future__ import annotations
import math
import weakref
from collections.abc import Hashable, Mapping
from numbers import Complex as Scalar, Real
from typing import TYPE_CHECKING
import numpy as np

from diffractix.backend import xp

from .ops import Op
from .relations import Relation, Comparison, SymbolicControlFlowError

if TYPE_CHECKING:
    from diffractix.elements import OpticalElement


class Node:
    """
    Base class for declarative nodes in the parameter AST.

    Comparisons between Nodes produce symbolic Comparison objects.
    """

    def __bool__(self):
        raise SymbolicControlFlowError("Symbolic values cannot be used as Python booleans.")

    def evaluate(self, bindings: Mapping[Hashable, object] = {},
        *, parameter_values: Mapping[int, object] | None = None,
    ):
        """Evaluate this graph with optional symbol and parameter values."""
        from .graph_utils import evaluate_ast
        return evaluate_ast(self, bindings=bindings, parameter_values=parameter_values)

    def has_variable_parameters(self, bindings: Mapping[Hashable, object] = {}) -> bool:
        """Check for variable Parameters after resolving Symbol bindings.

        Every Symbol must have an explicit binding, as required to establish
        whether its compile-time binding contains variable Parameters.
        """
        from .graph_utils import walk_ast

        has_variable = False
        for node in walk_ast((self,), bindings):
            if isinstance(node, Symbol) and node.key not in bindings:
                raise KeyError(f"Missing binding for Symbol key {node.key!r}.")
            if isinstance(node, Parameter) and node.is_variable:
                has_variable = True
        return has_variable

    @staticmethod
    def _make_literal(value: Scalar | np.ndarray) -> Literal:
        if isinstance(value, np.generic):
            value = value.item()
        return Literal(value)

    @staticmethod
    def _make_binary_op(op: Op, left: Node | Scalar, right: Node | Scalar) -> BinaryOp:
        if not isinstance(left, Node):
            left = Node._make_literal(left)
        if not isinstance(right, Node):
            right = Node._make_literal(right)
        return BinaryOp(op, left, right)

    @staticmethod
    def _make_unary_op(op: Op, operand: Node | Scalar) -> UnaryOp:
        if not isinstance(operand, Node):
            operand = Node._make_literal(operand)
        return UnaryOp(op, operand)

    @staticmethod
    def _make_comparison(relation: Relation, left: Node | Scalar, right: Node | Scalar) -> Comparison:
        if not isinstance(left, Node):
            left = Node._make_literal(left)

        if not isinstance(right, Node):
            right = Node._make_literal(right)

        return Comparison(relation, left, right)

    def __eq__(self, other: Node | Scalar) -> Comparison:
        return Node._make_comparison(Relation.EQ, self, other)

    def __le__(self, other: Node | Scalar) -> Comparison:
        return Node._make_comparison(Relation.LE, self, other)

    def __ge__(self, other: Node | Scalar) -> Comparison:
        return Node._make_comparison(Relation.GE, self, other)

    def __lt__(self, other: Node | Scalar) -> Comparison:
        return Node._make_comparison(Relation.LT, self, other)

    def __gt__(self, other: Node | Scalar) -> Comparison:
        return Node._make_comparison(Relation.GT, self, other)

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

    def sin(self) -> UnaryOp:
        return Node._make_unary_op(Op.SIN, self)

    def cos(self) -> UnaryOp:
        return Node._make_unary_op(Op.COS, self)

    def tan(self) -> UnaryOp:
        return Node._make_unary_op(Op.TAN, self)

    def sinh(self) -> UnaryOp:
        return Node._make_unary_op(Op.SINH, self)

    def cosh(self) -> UnaryOp:
        return Node._make_unary_op(Op.COSH, self)

    def tanh(self) -> UnaryOp:
        return Node._make_unary_op(Op.TANH, self)

    def sinc(self) -> UnaryOp:
        return Node._make_unary_op(Op.SINC, self)

    # -----------------
    # BINARY OPERATIONS
    # -----------------
    def __add__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, self, other)

    def __radd__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.ADD, other, self)

    def __sub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, self, other)

    def __rsub__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.SUB, other, self)

    def __mul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, self, other)

    def __rmul__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.MUL, other, self)

    def __truediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, self, other)

    def __rtruediv__(self, other: Node | Scalar) -> BinaryOp:
        return Node._make_binary_op(Op.DIV, other, self)

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
    def is_variable(self) -> bool:
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

    Allows the underlying node to be replaced without rebuilding expressions
    that reference the handle. Attribute access and assignment are forwarded
    to the underlying node.
    """

    def __init__(self, node: Node | None):
        self.node = node

    def __getattr__(self, name):
        if self.node is None:
            raise AttributeError(f"InputNode is empty and has no attribute '{name}'")
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
    Immutable numeric scalar or array embedded in an expression.

    Arrays are stored as owned, read-only NumPy copies. Zero-dimensional
    arrays retain the scalar normalization used by expression construction.

    Example:
        y = 2 * x

    The value 2 is represented by a Literal rather than a Parameter.
    """

    def __init__(self, value: Scalar | np.ndarray):
        if isinstance(value, Scalar):
            self._value = value
            return
        if isinstance(value, (np.ndarray, xp.ndarray)):
            try:
                value = np.array(value, copy=True)
            except (TypeError, ValueError) as error:
                raise TypeError("Literal value must be a concrete numeric scalar or array.") from error
            if value.dtype.kind not in "biufc":
                raise TypeError(f"Literal array must have a numeric dtype, got {value.dtype}.")
            if value.ndim == 0:
                value = value.item()
            else:
                value.setflags(write=False)
        else:
            raise TypeError("Literal value must be a numeric scalar or array.")

        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def is_variable(self) -> bool:
        return False

    def __repr__(self) -> str:
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
        bounds: tuple[float | None, float | None] | None = None,
        owner: "OpticalElement | None" = None,
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Parameter value must be a real scalar.")
        if math.isnan(value):
            raise ValueError(f"Parameter {name or '<parameter>'!r} value must not be NaN.")

        self.value = float(value)
        self.name = name
        self._is_variable = variable
        self.lower_bound, self.upper_bound = -np.inf, np.inf
        self.bounds = bounds
        self._owner = weakref.ref(owner) if owner is not None else None

    @property
    def is_variable(self) -> bool:
        return self._is_variable

    def variable(self, *, bounds=None):
        """Mark variable, optionally setting `bounds=(lower, upper)` (see `bounds`)."""
        if bounds is not None:
            self.bounds = bounds
        self._is_variable = True
        return self

    def fixed(self) -> Parameter:
        self._is_variable = False
        return self

    @staticmethod
    def _validate_bounds(lower_bound, upper_bound) -> None:
        """Validate a Parameter's numerical interval."""
        for name, bound in (
            ("lower_bound", lower_bound),
            ("upper_bound", upper_bound),
        ):
            if isinstance(bound, bool) or not isinstance(bound, Real):
                raise TypeError(f"{name} must be a real scalar.")
            if math.isnan(bound):
                raise ValueError(f"{name} must not be NaN.")

        if lower_bound > upper_bound:
            raise ValueError("lower_bound must be <= upper_bound.")

    @property
    def bounds(self) -> tuple[float, float]:
        """The interval `(lower, upper)`; infinities mean unbounded.

        Assign `(lower, upper)` to change it. A `None` endpoint, or `None` for the
        whole pair, keeps the current value; use `-np.inf`/`np.inf` to remove a bound.
        """
        return self.lower_bound, self.upper_bound

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            return
        try:
            lower_bound, upper_bound = bounds
        except (TypeError, ValueError):
            raise ValueError("bounds must be a (lower, upper) pair.") from None
        lower_bound = self.lower_bound if lower_bound is None else lower_bound
        upper_bound = self.upper_bound if upper_bound is None else upper_bound
        self._validate_bounds(lower_bound, upper_bound)

        if self.value < lower_bound or self.value > upper_bound:
            raise ValueError(
                f"Current value {self.value} lies outside bounds "
                f"[{lower_bound}, {upper_bound}]."
            )

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def owner(self):
        if self._owner is None:
            return None
        return self._owner()

    @property
    def full_name(self) -> str:
        owner = self.owner

        if owner is not None:
            label = getattr(owner, "label", None) or type(owner).__name__
            if self.name is not None:
                return f"{label}.{self.name}"
            return label

        return self.name or "<parameter>"

    def __repr__(self) -> str:
        status = "V" if self.is_variable else "F"
        return f"{self.full_name}={self.value:.4g}[{status}]"



class Symbol(Node):
    """Reference to an external value identified by a hashable key."""

    def __init__(self, key, *, default=None):
        try:
            hash(key)
        except TypeError:
            raise TypeError("Symbol key must be hashable.") from None

        self._key = key
        self._default_value = default

    def set_default(self, value):
        """Set the optional fallback value and return this Symbol."""
        self._default_value = value
        return self

    @property
    def key(self):
        return self._key

    @property
    def default(self):
        """The optional default value, or None when no default was provided."""
        return self._default_value

    @property
    def has_default(self):
        """Whether a non-None default is available."""
        return self._default_value is not None

    @property
    def is_variable(self) -> bool:
        return False

    @property
    def value(self):
        raise RuntimeError(
            f"{self!r} has no standalone value. "
            "Symbols are resolved from compile-time context or runtime bindings."
        )

    def __repr__(self) -> str:
        return f"Symbol({self.key!r})"


