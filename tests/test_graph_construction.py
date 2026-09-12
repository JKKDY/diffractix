import pytest
import autograd.numpy as np

from diffractix.graph.node import (
    BinaryOp,
    UnaryOp,
)

from diffractix.graph import (
    Comparison,
    Literal,
    Node,
    Parameter,
    Symbol,
    Relation,
    compile_ast,
    InputNode
)



# -----------------
# LITERAL SEMANTICS
# -----------------
def test_literal_math():
    """Literals should support all standard binary arithmetic operations."""
    a = Literal(10)
    b = Literal(5)

    assert (a + b).value == 15
    assert (a - b).value == 5
    assert (a * b).value == 50
    assert (a / b).value == 2
    assert (a ** 2).value == 100
    assert a.maximum(b).value == 10
    assert a.minimum(b).value == 5


def test_reverse_scalar_math():
    """Raw scalars should work on the left-hand side of Node operations."""
    x = Literal(4)

    assert (2 + x).value == 6
    assert (10 - x).value == 6
    assert (3 * x).value == 12
    assert (20 / x).value == 5
    assert (2 ** x).value == 16


def test_scalars_are_wrapped_as_literals():
    """
    Raw scalars used in expressions should become Literal nodes.

    They are mathematical constants in the expression, not design Parameters.
    """
    x = Parameter(4, name="x")
    expr = x * 2

    assert isinstance(expr, BinaryOp)
    assert expr.left is x
    assert isinstance(expr.right, Literal)
    assert expr.right.value == 2


def test_unary_operations():
    """Unary operators and functions should preserve operands and evaluate correctly."""
    x = Parameter(4, name="x")

    neg = -x
    pos = +x
    absolute = abs(-x)
    sigmoid = x.sigmoid()
    exponential = x.exp()
    logarithm = x.log()
    square_root = x.sqrt()

    assert isinstance(neg, UnaryOp)
    assert neg.operand is x

    assert neg.value == -4
    assert pos.value == 4
    assert absolute.value == 4
    assert sigmoid.value == pytest.approx(1 / (1 + np.exp(-4)))
    assert exponential.value == pytest.approx(np.exp(4))
    assert logarithm.value == pytest.approx(np.log(4))
    assert square_root.value == pytest.approx(2.0)


def test_sigmoid():
    x = Literal(0)

    assert x.sigmoid().value == pytest.approx(0.5)


def test_invalid_scalar_math():
    """Non-numeric operands should fail during AST construction."""
    x = Literal(1)

    with pytest.raises(TypeError):
        _ = x + "invalid"

    with pytest.raises(TypeError):
        _ = x + None


def test_literal_is_not_variable():
    x = Literal(10)

    assert x.is_variable is False


def test_literal_value_is_immutable():
    """Literal values are part of the expression itself and cannot be reassigned."""
    x = Literal(10)

    with pytest.raises(AttributeError):
        x.value = 20


def test_complex_literal():
    value = Literal(2 + 3j)

    assert value.value == 2 + 3j
    assert not value.is_variable


def test_complex_expression():
    x = Parameter(2.0)

    expr = -1j * x

    assert expr.value == -2j



# -------------------
# PARAMETER SEMANTICS
# -------------------
def test_parameter_is_fixed_by_default():
    """Parameters should be excluded from optimization unless explicitly made variable."""
    x = Parameter(10, name="x")

    assert x.is_variable is False


def test_parameter_variable_and_fixed_are_fluent():
    """Variable state setters should mutate in place and support method chaining."""
    x = Parameter(10, name="x")

    assert x.variable() is x
    assert x.is_variable is True

    assert x.fixed() is x
    assert x.is_variable is False


def test_parameter_constructor_can_make_variable():
    x = Parameter(10, name="x", variable=True)

    assert x.is_variable is True


def test_parameter_value_is_mutable():
    """
    Fixed Parameters remain mutable design values.

    'Fixed' only means excluded from the optimization variable vector.
    """
    x = Parameter(10, name="x")

    x.value = 20

    assert x.value == 20


def test_parameter_bounds():
    """Bounds should be stored fluently and reject an inverted interval."""
    x = Parameter(10, name="x")

    assert x.bound(5, 20) is x
    assert x.lower_bound == 5
    assert x.upper_bound == 20

    with pytest.raises(ValueError):
        x.bound(20, 5)


def test_parameter_without_name():
    """Standalone Parameters do not require names."""
    x = Parameter(10)

    assert x.name is None
    assert x.full_name == "<parameter>"


def test_expression_variable_state_propagates():
    """
    Expression variability should be derived dynamically from its dependencies.

    Changing a Parameter between fixed and variable must therefore propagate
    through already-created expressions.
    """
    x = Parameter(2, name="x")
    expr = 3 * x + 1

    assert expr.is_variable is False

    x.variable()
    assert expr.is_variable is True

    x.fixed()
    assert expr.is_variable is False


# ------------------------
# GRAPH IDENTITY SEMANTICS
# ------------------------
def test_equivalent_expressions_are_not_interned():
    """
    Structurally equivalent expressions are distinct graph objects.

    Deduplication is a compilation concern, not an AST construction rule.
    """
    x = Parameter(2, name="x")

    a = x + 1
    b = x + 1

    assert a is not b


def test_explicit_expression_sharing_is_preserved():
    """Explicitly reusing a node should preserve graph sharing by identity."""
    x = Parameter(2, name="x")

    shared = x + 1
    a = shared * 2
    b = shared * 3

    assert a.left is shared
    assert b.left is shared


# -------------------
# INPUTNODE SEMANTICS
# -------------------
def test_input_node_forwards_attributes():
    """InputNode should transparently forward reads and writes to its target."""
    x = Parameter(10, name="x")
    handle = InputNode(x)

    assert handle.value == 10
    assert handle.name == "x"
    assert handle.is_variable is False

    handle.value = 20

    assert x.value == 20


def test_input_node_forwards_parameter_methods():
    """Parameter methods reached through an InputNode should affect the target."""
    x = Parameter(10, name="x")
    handle = InputNode(x)

    handle.variable()

    assert x.is_variable is True
    assert handle.is_variable is True

    handle.fixed()

    assert x.is_variable is False


def test_input_node_is_preserved_in_expression():
    """
    Expressions should reference the InputNode handle itself.

    This is required so later hot-swapping changes the declarative graph seen
    by already-created downstream expressions.
    """
    handle = InputNode(Parameter(10))
    expr = handle + 5

    assert expr.left is handle
    assert expr.value == 15


def test_input_node_hot_swap_updates_existing_expression():
    """Replacing an InputNode target should update existing expressions."""
    handle = InputNode(Parameter(10))
    expr = handle * 2

    assert expr.value == 20

    handle.node = Parameter(5)

    assert expr.value == 10


def test_input_node_can_swap_to_expression():
    """
    An InputNode may point to an arbitrary AST expression, not only a Parameter.
    """
    handle = InputNode(Literal(10))
    result = handle + 5

    z = Parameter(3, name="z").variable()
    handle.node = z * 2

    assert result.value == 11
    assert result.is_variable is True


def test_empty_input_node():
    """Empty handles may exist temporarily but cannot forward attributes."""
    handle = InputNode(None)

    assert handle.is_variable is False

    with pytest.raises(AttributeError):
        _ = handle.value

    with pytest.raises(AttributeError):
        handle.value = 10


# ----------------
# SYMBOL SEMANTICS
# ----------------
def test_symbol_requires_hashable_key():
    """Symbol keys must be usable for context and runtime binding lookup."""
    with pytest.raises(TypeError, match="must be hashable"):
        Symbol(["not", "hashable"])


def test_symbol_has_no_standalone_value():
    """A Symbol obtains its value only during compilation or evaluation."""
    symbol = Symbol(("result", 42, "w"))

    assert symbol.key == ("result", 42, "w")
    assert symbol.is_variable is False

    with pytest.raises(RuntimeError):
        _ = symbol.value


def test_symbols_are_not_interned():
    """
    Equal Symbol keys need not imply object identity.

    Key-based resolution happens through the compilation context.
    """
    a = Symbol("ambient_n")
    b = Symbol("ambient_n")

    assert a is not b


# -------------------
# RELATION CREATION
# -------------------
def test_equality_creates_comparison():
    x = Parameter(1.0)
    comparison = x == 2.0

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.EQ
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_less_equal_creates_comparison():
    x = Parameter(1.0)
    comparison = x <= 2.0

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.LE
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_greater_equal_creates_comparison():
    x = Parameter(1.0)
    comparison = x >= 2.0

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.GE
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_less_than_creates_comparison():
    x = Parameter(1.0)
    comparison = x < 2.0

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.LT
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_greater_than_creates_comparison():
    x = Parameter(1.0)
    comparison = x > 2.0

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.GT
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


# ----------------
# NODE OPERANDS
# ----------------
def test_comparison_preserves_node_operands():
    x = Parameter(1.0)
    y = Parameter(2.0)

    comparison = x <= y

    assert comparison.left is x
    assert comparison.right is y


def test_comparison_accepts_expression_operands():
    x = Parameter(1.0)
    y = Parameter(2.0)

    expression = x + y
    comparison = expression == 3.0

    assert comparison.relation is Relation.EQ
    assert comparison.left is expression
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 3.0


def test_comparison_between_expressions():
    x = Parameter(1.0)
    y = Parameter(2.0)

    left = x + 1.0
    right = 2.0 * y
    comparison = left >= right

    assert comparison.relation is Relation.GE
    assert comparison.left is left
    assert comparison.right is right


# ---------------------
# REFLECTED COMPARISON
# ---------------------

def test_scalar_less_than_node_uses_reflected_relation():
    x = Parameter(1.0)

    comparison = 0.0 < x

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.GT
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 0.0


def test_scalar_less_equal_node_uses_reflected_relation():
    x = Parameter(1.0)

    comparison = 0.0 <= x

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.GE
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 0.0


def test_scalar_greater_than_node_uses_reflected_relation():
    x = Parameter(1.0)

    comparison = 2.0 > x

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.LT
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_scalar_greater_equal_node_uses_reflected_relation():
    x = Parameter(1.0)

    comparison = 2.0 >= x

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.LE
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


def test_scalar_equality_node_creates_comparison():
    x = Parameter(1.0)

    comparison = 1.0 == x

    assert isinstance(comparison, Comparison)
    assert comparison.relation is Relation.EQ
    assert comparison.left is x
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 1.0


# -------------------
# BOOLEAN SEMANTICS
# -------------------
def test_symbolic_comparison_has_no_truth_value():
    x = Parameter(1.0)
    comparison = x == 1.0

    with pytest.raises(
        TypeError,
        match="cannot be used as booleans",
    ):
        bool(comparison)


def test_symbolic_comparison_fails_in_if_statement():
    x = Parameter(1.0)

    with pytest.raises(
        TypeError,
        match="cannot be used as booleans",
    ):
        if x == 1.0:
            pass


# ------------------
# TERMINAL SEMANTICS
# ------------------

def test_comparison_is_not_graph_node():
    x = Parameter(1.0)
    comparison = x == 1.0

    assert not isinstance(comparison, Node)


def test_comparison_cannot_participate_in_arithmetic():
    x = Parameter(1.0)
    comparison = x == 1.0

    with pytest.raises(TypeError):
        comparison + 1.0


def test_comparison_cannot_be_compiled_as_ast_root():
    x = Parameter(1.0)
    comparison = x == 1.0

    with pytest.raises(
        TypeError,
        match="AST root must be a Node",
    ):
        compile_ast((comparison,))


# -----------------
# IDENTITY SAFETY
# -----------------
def test_nodes_are_unhashable_after_symbolic_equality():
    x = Parameter(1.0)

    with pytest.raises(TypeError):
        hash(x)


def test_distinct_equal_valued_parameters_remain_distinct():
    x = Parameter(1.0, name="x").variable()
    y = Parameter(1.0, name="x").variable()

    graph = compile_ast((x + y,))

    assert len(graph.variables) == 2
    assert graph.variables[0] is x
    assert graph.variables[1] is y


# --------------
# REPRESENTATION
# --------------
@pytest.mark.parametrize(
    ("relation", "symbol"),
    [
        (Relation.EQ, "=="),
        (Relation.LE, "<="),
        (Relation.GE, ">="),
        (Relation.LT, "<"),
        (Relation.GT, ">"),
    ],
)
def test_relation_values(relation, symbol):
    assert relation.value == symbol


def test_comparison_repr_contains_relation():
    x = Parameter(1.0, name="x")
    comparison = x <= 2.0

    assert "<=" in repr(comparison)
