import pytest

from diffractix.graph.node import (
    Node,
    Parameter,
    Symbol,
    InputNode,
)
from diffractix.graph.utils import (
    ASTCycleError,
    UnresolvedInputError,
    UnsupportedNodeError,
    walk_ast,
    collect_variables,
    clone_ast,
)


# ------------------
# VARIABLE DISCOVERY
# ------------------
def test_collect_variables_filters_fixed_parameters():
    """Only Parameters marked variable should appear in the variable vector."""
    variable = Parameter(1, name="variable").variable()
    fixed = Parameter(2, name="fixed")

    variables = collect_variables([variable, fixed])

    assert variables == (variable,)


def test_collect_variables_preserves_order():
    """
    Variable ordering should follow deterministic graph traversal order.

    This order defines how external variable values map back to Parameters.
    """
    a = Parameter(1, name="a").variable()
    b = Parameter(2, name="b").variable()
    expr = a * b

    assert collect_variables([expr]) == (a, b)


def test_collect_variables_deduplicates_by_identity():
    """
    Reusing the same Parameter object must produce only one variable entry.
    """
    x = Parameter(2, name="x").variable()
    expr = x * x + x

    assert collect_variables([expr]) == (x,)


def test_distinct_parameters_are_not_deduplicated():
    """
    Parameter identity defines independent variables.

    Equal names and values must not cause distinct Parameter objects to merge.
    """
    a = Parameter(1, name="x").variable()
    b = Parameter(1, name="x").variable()

    variables = collect_variables([a + b])

    assert variables == (a, b)


def test_collect_variables_through_input_node():
    """Variable discovery should follow InputNode indirection."""
    x = Parameter(2, name="x").variable()
    handle = InputNode(x)

    assert collect_variables([handle * 2]) == (x,)


def test_collect_variables_through_bound_symbol():
    """
    Variable discovery should traverse AST nodes injected through context.
    """
    x = Parameter(2, name="x").variable()
    var = Symbol("external")

    variables = collect_variables(
        [var * 2],
        context={"external": x},
    )

    assert variables == (x,)


def test_multiple_symbols_can_resolve_to_same_parameter():
    """
    Multiple context references to the same Parameter still represent one variable.
    """
    a = Symbol("shared")
    b = Symbol("shared")
    x = Parameter(2, name="x").variable()

    variables = collect_variables(
        [a + b],
        context={"shared": x},
    )

    assert variables == (x,)


def test_scalar_context_value_adds_no_variable():
    """Scalar context values are terminal and introduce no variable Parameters."""
    var = Symbol("ambient_n")

    variables = collect_variables(
        [var * 2],
        context={"ambient_n": 1.5},
    )

    assert variables == ()


def test_collect_variables_treats_unbound_symbol_as_external_leaf():
    """An unbound Symbol introduces no Parameter optimization variables."""
    symbol = Symbol("missing")

    assert collect_variables([symbol]) == ()


# ----------------
# GRAPH TRAVERSAL
# ----------------
def test_empty_input_node_fails_traversal():
    """An empty InputNode is unresolved and cannot be traversed."""
    handle = InputNode(None)

    with pytest.raises(UnresolvedInputError):
        tuple(walk_ast([handle]))


def test_cycle_through_input_nodes_is_detected():
    """
    Cycles introduced through mutable InputNodes must be detected explicitly.

    This avoids relying on Python's recursion limit to identify invalid graphs.
    """
    a = InputNode(None)
    b = InputNode(None)

    a.node = b + 1
    b.node = a + 1

    with pytest.raises(ASTCycleError):
        tuple(walk_ast([a]))


def test_cycle_through_context_is_detected():
    """Context resolution may also introduce cycles and must be checked."""
    var = Symbol("self")

    with pytest.raises(ASTCycleError):
        tuple(walk_ast([var], {"self": var}))


def test_non_node_root_is_rejected():
    """AST traversal should only accept Node objects as roots."""
    with pytest.raises(TypeError):
        tuple(walk_ast([1]))


def test_unknown_node_type_is_rejected():
    """
    Traversal should fail explicitly for unsupported Node subclasses.

    Silently treating unknown nodes as leaves could hide incomplete graph logic.
    """
    class UnknownNode(Node):
        pass

    with pytest.raises(UnsupportedNodeError):
        tuple(walk_ast([UnknownNode()]))


# -------------
# AST CLONING
# -------------
def test_clone_ast_creates_independent_nodes():
    """
    Cloning should create a structurally equivalent but independent graph.
    """
    x = Parameter(2, name="x").variable()
    expr = x * 2

    cloned_expr, = clone_ast([expr])

    assert cloned_expr is not expr
    assert cloned_expr.left is not x
    assert cloned_expr.left.value == x.value
    assert cloned_expr.left.name == x.name
    assert cloned_expr.left.is_variable == x.is_variable


def test_clone_ast_preserves_explicit_sharing():
    """
    Explicit sharing in the source graph must remain sharing in the clone.

    A shared source node should be cloned once and referenced from all
    corresponding cloned branches.
    """
    x = Parameter(2, name="x").variable()
    shared = x + 1

    a = shared * 2
    b = shared * 3

    cloned_a, cloned_b = clone_ast([a, b])

    assert cloned_a.left is cloned_b.left
    assert cloned_a.left is not shared


def test_clone_ast_does_not_merge_equivalent_nodes():
    """
    Cloning preserves graph identity semantics.

    Structurally equivalent but distinct source nodes must remain distinct.
    """
    x = Parameter(2, name="x")

    a = x + 1
    b = x + 1

    cloned_a, cloned_b = clone_ast([a, b])

    assert cloned_a is not cloned_b


def test_clone_ast_clones_input_handle():
    """
    InputNode handles and their targets should both be independent in the clone.
    """
    handle = InputNode(Parameter(2, name="x"))

    cloned_handle, = clone_ast([handle])

    assert cloned_handle is not handle
    assert cloned_handle.node is not handle.node
    assert cloned_handle.node.value == handle.node.value


def test_clone_ast_clones_symbol_without_resolving_it():
    """Generic Symbols retain their keys when structurally cloned."""
    key = ("result", 42, "w")
    symbol = Symbol(key)

    cloned, = clone_ast([symbol])

    assert cloned is not symbol
    assert isinstance(cloned, Symbol)
    assert cloned.key == key


def test_clone_ast_detects_cycle():
    """Invalid cyclic graphs must not be cloneable."""
    handle = InputNode(None)
    handle.node = handle + 1

    with pytest.raises(ASTCycleError):
        clone_ast([handle])
