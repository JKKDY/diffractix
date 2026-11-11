import operator

import pytest
import numpy as np

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
    SymbolicControlFlowError,
    compile_graph,
    InputNode
)


# ----------------
# VARIABLE BOUNDS
# ----------------
def test_bounds_have_one_spelling_in_constructor_variable_and_property():
    constructed = Parameter(5e-3, bounds=(4e-3, 10e-3))
    marked = Parameter(5e-3)
    assert marked.variable(bounds=(4e-3, 10e-3)) is marked and marked.is_variable
    assigned = Parameter(5e-3)
    assigned.bounds = (4e-3, 10e-3)
    assert constructed.bounds == marked.bounds == assigned.bounds == (4e-3, 10e-3)
    assert not constructed.is_variable and not assigned.is_variable
    assert (marked.lower_bound, marked.upper_bound) == (4e-3, 10e-3)
    assert Parameter(5e-3).bounds == Parameter(5e-3, bounds=(-np.inf, np.inf)).bounds == (-np.inf, np.inf)


@pytest.mark.parametrize("bounds, expected", [
    ((None, 2.0), (0.5, 2.0)),
    ((0.0, None), (0.0, 1.5)),
    ((None, None), (0.5, 1.5)),
    (None, (0.5, 1.5)),
    ((-np.inf, np.inf), (-np.inf, np.inf)),
])
def test_none_keeps_the_current_endpoint_and_infinities_remove_it(bounds, expected):
    parameter = Parameter(1.0, bounds=(0.5, 1.5))
    parameter.variable(bounds=bounds)
    assert parameter.bounds == expected
    other = Parameter(1.0, bounds=(0.5, 1.5))
    other.bounds = bounds
    assert other.bounds == expected


def test_variable_without_bounds_keeps_existing_bounds():
    parameter = Parameter(1.0, bounds=(0.0, 2.0)).variable()
    assert parameter.is_variable and parameter.bounds == (0.0, 2.0)


def test_variable_accepts_bounds_only_by_keyword_and_has_no_endpoint_arguments():
    parameter = Parameter(1.0)
    for call in (lambda: parameter.variable(0.0, 2.0), lambda: parameter.variable(lower_bound=0.0),
                 lambda: parameter.variable(upper_bound=2.0), lambda: Parameter(1.0, lower_bound=0.0)):
        with pytest.raises(TypeError):
            call()
    assert not hasattr(parameter, "bound")


@pytest.mark.parametrize("bounds, error", [
    ((2.0, 0.0), ValueError),                                   # inverted
    ((1.2, 1.4), ValueError), ((0.6, 0.8), ValueError),         # current value outside
    ((True, 1.5), TypeError), ((0.5, "invalid"), TypeError),    # not real numbers
    ((np.nan, 1.5), ValueError), ((None, np.nan), ValueError),
    ((0.5,), ValueError), ((0.5, 1.0, 1.5), ValueError), (5.0, ValueError),   # not a pair
])
def test_bounds_validation_is_atomic_for_every_spelling(bounds, error):
    parameter = Parameter(1.0, bounds=(0.5, 1.5))
    with pytest.raises(error):
        parameter.variable(bounds=bounds)
    with pytest.raises(error):
        parameter.bounds = bounds
    with pytest.raises(error):
        Parameter(1.0, bounds=bounds)
    assert not parameter.is_variable
    assert parameter.bounds == (0.5, 1.5)


# ---------------------------
# LITERALS AND SCALAR OPERANDS
# ---------------------------
@pytest.mark.parametrize("node, expected", [
    (Literal(2), 2.0), (Parameter(2), 2.0),
    (Parameter(2) * 3, 6.0), (InputNode(Parameter(2)), 2.0),
])
def test_symbolic_nodes_cannot_be_implicitly_converted_to_float(node, expected):
    with pytest.raises(TypeError):
        float(node)
    assert float(node.value) == expected


@pytest.mark.parametrize("value", [np.int16(2), np.float32(2.5), np.complex64(2 + 1j)])
def test_numpy_numeric_scalars_are_valid_literals_and_expression_constants(value):
    assert Literal(value).value == value
    assert (Parameter(1) + value).value == 1 + value


@pytest.mark.parametrize("value", [np.int16(2), np.float32(2.5)])
def test_numpy_real_scalars_are_valid_parameters(value):
    assert Parameter(value).value == value


def test_scalar_array_constant_is_independent_of_the_original_array():
    constant = np.array(2.0)
    expression = Parameter(1) + constant
    constant[...] = 99
    assert expression.value == 3.0


@pytest.mark.parametrize("value", [np.array([2.0]), np.array([[2.0]])])
def test_array_constants_are_wrapped_as_owned_read_only_literals(value):
    expression = Parameter(1) + value
    assert isinstance(expression.right, Literal)
    np.testing.assert_array_equal(expression.right.value, value)
    assert expression.right.value.shape == value.shape
    assert expression.right.value.dtype == value.dtype
    assert not np.shares_memory(expression.right.value, value)
    assert not expression.right.value.flags.writeable


def test_scalar_like_object_is_not_implicitly_normalized():
    class ScalarLike:
        shape = ()

        def item(self):
            raise AssertionError("Unrecognized objects must not be converted")

    with pytest.raises(TypeError, match="numeric scalar"):
        Parameter(1) + ScalarLike()


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


@pytest.mark.parametrize("build", [
    lambda x: -x, lambda x: +x, abs, lambda x: x.sigmoid(), lambda x: x.exp(), lambda x: x.log(),
    lambda x: x.sqrt(), lambda x: x.sin(), lambda x: x.cos(), lambda x: x.tan(),
    lambda x: x.sinh(), lambda x: x.cosh(), lambda x: x.tanh(), lambda x: x.sinc(),
])
def test_unary_operations_build_nodes_that_keep_their_operand(build):
    """Values and derivatives of every operation are covered in test_operations.py."""
    x = Parameter(4, name="x")
    node = build(x)
    assert isinstance(node, UnaryOp)
    assert node.operand is x

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

    expression = 2 * x
    x.value = 20
    assert expression.value == 40


def test_parameter_constructor_can_make_variable():
    x = Parameter(10, name="x", variable=True)

    assert x.is_variable is True


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


def test_symbol_none_default_is_unset():
    absent = Symbol("ambient_n")
    explicit_none = Symbol("ambient_n", default=None)

    assert absent.default is None
    assert absent.has_default is False
    assert explicit_none.default is None
    assert explicit_none.has_default is False


def test_symbol_set_default_updates_value_and_returns_symbol():
    symbol = Symbol("ambient_n")

    assert symbol.set_default(1.5) is symbol
    assert symbol.default == 1.5
    assert symbol.has_default is True

    assert symbol.set_default(None) is symbol
    assert symbol.default is None
    assert symbol.has_default is False


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
@pytest.mark.parametrize("operation,relation", [
    pytest.param(operator.eq, Relation.EQ, id="eq"),
    pytest.param(operator.le, Relation.LE, id="le"),
    pytest.param(operator.ge, Relation.GE, id="ge"),
    pytest.param(operator.lt, Relation.LT, id="lt"),
    pytest.param(operator.gt, Relation.GT, id="gt"),
])
def test_scalar_comparison_preserves_operands_and_relation(operation, relation):
    parameter = Parameter(1.0)
    comparison = operation(parameter, 2.0)

    assert isinstance(comparison, Comparison)
    assert comparison.relation is relation
    assert comparison.left is parameter
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == 2.0


# ----------------
# NODE OPERANDS
# ----------------
@pytest.mark.parametrize("make_left, make_right, relation", [
    pytest.param(lambda x, y: x, lambda x, y: y, operator.le, id="parameters"),
    pytest.param(lambda x, y: x + y, lambda x, y: 3.0, operator.eq, id="expression-and-scalar"),
    pytest.param(lambda x, y: x + 1.0, lambda x, y: 2.0 * y, operator.ge, id="two-expressions"),
])
def test_comparison_keeps_node_and_expression_operands_by_identity(make_left, make_right, relation):
    x, y = Parameter(1.0), Parameter(2.0)
    left, right = make_left(x, y), make_right(x, y)
    comparison = relation(left, right)

    assert isinstance(comparison, Comparison)
    assert comparison.left is left
    if isinstance(right, Node):
        assert comparison.right is right
    else:
        assert isinstance(comparison.right, Literal) and comparison.right.value == right


# ---------------------
# REFLECTED COMPARISON
# ---------------------

@pytest.mark.parametrize("operation,scalar,relation", [
    pytest.param(operator.lt, 0.0, Relation.GT, id="lt"),
    pytest.param(operator.le, 0.0, Relation.GE, id="le"),
    pytest.param(operator.gt, 2.0, Relation.LT, id="gt"),
    pytest.param(operator.ge, 2.0, Relation.LE, id="ge"),
    pytest.param(operator.eq, 1.0, Relation.EQ, id="eq"),
])
def test_reflected_scalar_comparison_preserves_operands_and_relation(operation, scalar, relation):
    parameter = Parameter(1.0)
    comparison = operation(scalar, parameter)

    assert isinstance(comparison, Comparison)
    assert comparison.relation is relation
    assert comparison.left is parameter
    assert isinstance(comparison.right, Literal)
    assert comparison.right.value == scalar


# -------------------
# BOOLEAN SEMANTICS
# -------------------
def test_symbolic_comparison_has_no_truth_value_in_bool_or_if():
    x = Parameter(1.0)

    with pytest.raises(SymbolicControlFlowError, match="cannot be used as Python booleans"):
        bool(x == 1.0)
    with pytest.raises(SymbolicControlFlowError, match="cannot be used as Python booleans"):
        if x == 1.0:
            pass


def test_symbolic_control_flow_error_is_a_type_error():
    assert issubclass(SymbolicControlFlowError, TypeError)


@pytest.mark.parametrize(
    "node",
    [Literal(1.0), Parameter(1.0), Symbol("runtime")],
)
def test_symbolic_nodes_reject_python_boolean_conversion(node):
    with pytest.raises(
        SymbolicControlFlowError,
        match="Symbolic values cannot be used as Python booleans",
    ):
        bool(node)


def test_unrelated_type_errors_are_not_symbolic_control_flow_errors():
    with pytest.raises(TypeError) as error:
        _ = Literal(1) + "invalid"

    assert not isinstance(error.value, SymbolicControlFlowError)


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
        compile_graph((comparison,))


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

    graph = compile_graph((x + y,))

    assert graph.parameter_indices == {id(x): 0, id(y): 1}


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
def test_comparison_repr_contains_unambiguous_relation(relation, symbol):
    x = Parameter(1.0, name="x")
    comparison = Comparison(relation, x, Literal(2.0))

    assert f" {symbol} " in repr(comparison)


@pytest.mark.parametrize("bounds", [{}, {"bounds": (0, 1)}])
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("name", [None, "invalid"])
def test_parameter_rejects_nan_at_construction(bounds, variable, name):
    with pytest.raises(ValueError, match="NaN") as error:
        Parameter(np.nan, name=name, variable=variable, **bounds)
    assert repr(name or "<parameter>") in str(error.value)


def test_full_name_combines_the_owner_label_with_the_parameter_name():
    """Used in messages and reports: owner label, then the field name when there is one."""
    from diffractix.elements import ThinLens
    lens = ThinLens(f=0.1, label="Objective")

    assert lens.f.node.full_name == "Objective.f"
    assert Parameter(1.0, owner=lens).full_name == "Objective"
    assert Parameter(1.0, name="spacing").full_name == "spacing"
    assert Parameter(1.0).full_name == "<parameter>"
    assert repr(Parameter(1.5, name="spacing").variable()) == "spacing=1.5[V]"
    assert repr(Parameter(1.5, name="spacing")) == "spacing=1.5[F]"
