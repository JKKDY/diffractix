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
    """Base class for a node in diffractix's expression graph.

    Arithmetic between Nodes and numbers builds a graph rather than computing a
    result immediately: ``2 * x + 1`` returns a new Node. `Literal`, `Parameter`,
    `Symbol`, `InputNode`, `UnaryOp` and `BinaryOp` are its concrete subclasses.
    Comparisons such as ``x <= 1`` return a `Comparison`, for use in constraints,
    not a Python bool.

    Notes
    -----
    Beyond the usual operators, a Node supports ``sigmoid``, ``exp``, ``log``,
    ``sqrt``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``, ``tanh``, ``sinc``
    (unnormalized: ``sinc(x) = sin(x) / x``), and ``maximum``/``minimum`` with
    another Node or number.
    """

    def __bool__(self):
        raise SymbolicControlFlowError("Symbolic values cannot be used as Python booleans.")

    def evaluate(self, *,
        bindings: Mapping[Hashable, object] = {},
        parameter_values: Mapping[int, object] | None = None,
    ):
        """Evaluate this graph.

        Parameters
        ----------
        bindings : Mapping, optional
            The value of each Symbol in the graph, keyed by `Symbol.key`, such as
            ``{"wavelength": 1.064e-6}``. A value can also be an expression,
            such as ``{"k": 2 * p}``, which is evaluated as part of the graph.
            Defaults to ``{}``.
        parameter_values : Mapping, optional
            Values to use instead of the Parameters' current values, keyed by
            ``id(parameter)``. If given, it must cover every Parameter in the
            graph. Defaults to ``None``: each Parameter's current `value` is used.

        Returns
        -------
        float or array_like
            The result.

        Raises
        ------
        KeyError
            If a Symbol in the graph has no binding, or
            `parameter_values` is given and misses a Parameter in the graph.

        Examples
        --------
        >>> from diffractix import Parameter, Symbol
        >>> p = Parameter(2.0)
        >>> expression = p * Symbol("k")
        >>> float(expression.evaluate(bindings={"k": 3.0}))
        6.0
        >>> float(expression.evaluate(bindings={"k": 3.0}, parameter_values={id(p): 5.0}))
        15.0
        """
        from .graph_utils import evaluate_ast
        return evaluate_ast(self, bindings=bindings, parameter_values=parameter_values)

    def has_variable_parameters(self, bindings: Mapping[Hashable, object] = {}) -> bool:
        """Check whether this graph contains a variable Parameter.

        Symbols are resolved recursively through `bindings` along the way.

        Parameters
        ----------
        bindings : Mapping, optional
            Values for Symbols, by key, resolved recursively. Defaults to ``{}``.

        Returns
        -------
        bool
            Whether a variable Parameter was reached.

        Raises
        ------
        KeyError
            If a reached Symbol has no entry in `bindings`.
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
    """A binary operation, such as ``a + b``, built automatically by arithmetic
    between Nodes."""

    def __init__(self, op: Op, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op.unicode} {self.right})"

    @property
    def value(self):
        """Value: the operation applied to the operands' current values."""
        return self.op.func(self.left.value, self.right.value)

    @property
    def is_variable(self) -> bool:
        """Whether either operand reaches a variable Parameter."""
        return self.left.is_variable or self.right.is_variable


class UnaryOp(Node):
    """A unary operation, such as ``-a`` or ``a.sqrt()``, built automatically by
    arithmetic on a Node."""

    def __init__(self, op: Op, operand: Node):
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"{self.op.unicode}({self.operand})"

    @property
    def value(self):
        """Value: the operation applied to the operand's current value."""
        return self.op.func(self.operand.value)

    @property
    def is_variable(self):
        """Whether the operand reaches a variable Parameter."""
        return self.operand.is_variable


class InputNode(Node):
    """Stable handle to another node, whose underlying node can be replaced
    without rebuilding expressions that reference the handle.

    A `Node`-annotated field on an element, composite or state is exposed as one
    of these: assigning a number creates a `Parameter` inside it; assigning an
    expression stores that expression. Attribute access and assignment on the
    handle are forwarded to the underlying node, so ``lens.f.value`` and
    ``lens.f.variable()`` work directly, and ``lens.f = 0.2`` later replaces what
    the handle points to.

    Parameters
    ----------
    node : Node or None
        The underlying node. ``None`` for an empty handle, which has no
        attributes and cannot be evaluated or compiled.
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
        """Whether the underlying node reaches a variable Parameter; ``False`` if
        the handle is empty."""
        return self.node.is_variable if self.node is not None else False


class Literal(Node):
    """An immutable numeric constant embedded in an expression.

    An ordinary number used as an operand in an expression, such as the ``2`` in
    ``2 * x``, is wrapped in one of these automatically.

    Parameters
    ----------
    value : complex or array_like.
        An array is stored as an owned, read-only copy; a
        zero-dimensional array is reduced to a plain scalar.

    Raises
    ------
    TypeError
        If `value` is not a numeric scalar or array, or an array does not have a
        numeric dtype.
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
        """Value: the constant."""
        return self._value

    @property
    def is_variable(self) -> bool:
        """Always ``False``."""
        return False

    def __repr__(self) -> str:
        return repr(self._value)


class Parameter(Node):
    """A mutable, real-valued scalar design quantity.

    Parameters are fixed by default i.e. excluded from optimization.
    Mark one variable with :meth:`variable` to make it an optimizable degree of freedom.
    A Parameter may stand alone or come from an `OpticalBase` field that was
    assigned a number.

    Parameters
    ----------
    value : float
        Initial value. Must not be NaN.
    name : str, optional
        Name used in `full_name` and in messages. Defaults to ``None``: no name.
    variable : bool, optional
        Whether this Parameter is variable. Defaults to ``False``.
    bounds : tuple of (float or None, float or None), optional
        ``(lower, upper)``; see `bounds`. Defaults to unbounded.
    owner : OpticalElement, optional
        The element this Parameter belongs to. Only used to name it, in
        `full_name` and so in its repr and error messages. Held weakly.
        Defaults to ``None``: it stands alone.

    Raises
    ------
    TypeError
        If `value` is not a real scalar.
    ValueError
        If `value` is NaN.

    Examples
    --------
    >>> from diffractix import Parameter
    >>> p = Parameter(0.5).variable(bounds=(0.0, 1.0))
    >>> p.is_variable, p.bounds
    (True, (0.0, 1.0))
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
        if bounds is not None:
            self.bounds = bounds
        self._owner = weakref.ref(owner) if owner is not None else None

    @property
    def is_variable(self) -> bool:
        """Whether this Parameter is currently a degree of freedom."""
        return self._is_variable

    def variable(self, *, bounds=None):
        """Mark this Parameter variable.

        Parameters
        ----------
        bounds : tuple of (float or None, float or None), optional
            New bounds; see :attr:`bounds`. Defaults to leaving the current bounds.

        Returns
        -------
        Parameter
            ``self``, so calls can be chained.
        """
        if bounds is not None:
            self.bounds = bounds
        self._is_variable = True
        return self

    def fixed(self) -> Parameter:
        """Mark this Parameter fixed and return it.

        Returns
        -------
        Parameter
            ``self``, so calls can be chained.
        """
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
        """Bounds: the interval ``(lower, upper)`` this Parameter's value must lie
        in; infinities mean unbounded.

        Assign a ``(lower, upper)`` pair to change it. A ``None`` endpoint keeps
        the current bound; use ``-inf``/``inf`` to remove one.

        Raises
        ------
        ValueError
            If the assigned value is not a ``(lower, upper)`` pair, a bound is
            NaN, the bounds are not ordered, or the current value lies outside
            them.
        TypeError
            If a bound is not a real scalar.
        """
        return self.lower_bound, self.upper_bound

    @bounds.setter
    def bounds(self, bounds):
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
        """Owner: the element this Parameter belongs to, or ``None`` if it stands
        alone or the element has been garbage-collected."""
        if self._owner is None:
            return None
        return self._owner()

    @property
    def full_name(self) -> str:
        """Full name: the owner's label (or type name), joined to this
        Parameter's name by a dot if it has one, else just the label; just the
        name, or ``"<parameter>"``, if it has no owner."""
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
    """Reference to an external value, resolved from compile-time bindings or
    runtime bindings.

    Elements can depend on values that are only known elsewhere, such as the system's
    ambient index or wavelength, by using a Symbol with an agreed-on key; the
    System or Solver supplies the value through bindings at compile time, or the
    value is supplied at runtime (e.g. traced Constraints).

    Parameters
    ----------
    key : hashable
        Identifies this Symbol; Symbols with the same key are interchangeable.

    Raises
    ------
    TypeError
        If `key` is not hashable.
    """

    def __init__(self, key):
        try:
            hash(key)
        except TypeError:
            raise TypeError("Symbol key must be hashable.") from None

        self._key = key

    @property
    def key(self):
        """Key: identifies this Symbol."""
        return self._key

    @property
    def is_variable(self) -> bool:
        """Always ``False``."""
        return False

    @property
    def value(self):
        """A Symbol has no standalone value; resolve it through bindings instead.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError(
            f"{self!r} has no standalone value. "
            "Symbols are resolved from compile-time context or runtime bindings."
        )

    def __repr__(self) -> str:
        return f"Symbol({self.key!r})"


