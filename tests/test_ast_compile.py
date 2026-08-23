import numpy as np
import pytest
from autograd import grad
import autograd.numpy as anp

from diffractix.graph.ast import Literal, Parameter, SystemVar, InputNode
from diffractix.graph.compile import CompiledAST, compile_ast, Opcode
from diffractix.graph.utils import (
    ASTCycleError,
    UnresolvedInputError,
    UnresolvedSystemVarError,
)


# -----------------
# BASIC COMPILATION
# -----------------
def test_compile_basic_expression():
    """A compiled AST should expose variables in order and evaluate correctly."""
    m = Parameter(2, name="m").variable()
    x = Parameter(3, name="x").variable()
    c = Parameter(1, name="c")

    compiled = compile_ast([m * x + c])

    assert isinstance(compiled, CompiledAST)
    assert compiled.variables == (m, x)
    np.testing.assert_array_equal(compiled.initial_values, [2, 3])

    result = compiled.evaluate(np.array([10.0, 5.0]))
    np.testing.assert_array_equal(result, [51.0])


def test_compiled_ast_is_callable():
    """Calling CompiledAST directly should delegate to evaluate()."""
    x = Parameter(2, name="x").variable()
    compiled = compile_ast([x * 3])

    np.testing.assert_array_equal(
        compiled(np.array([4.0])),
        compiled.evaluate(np.array([4.0])),
    )


def test_root_order_is_preserved():
    """
    Root order defines output order.

    Variable discovery should follow the same deterministic left-to-right
    traversal of the supplied roots.
    """
    a = Parameter(1, name="a").variable()
    b = Parameter(2, name="b").variable()

    compiled = compile_ast([b, a])

    assert compiled.variables == (b, a)

    result = compiled.evaluate(np.array([20.0, 10.0]))
    np.testing.assert_array_equal(result, [20.0, 10.0])


def test_compile_constant_expression():
    """An AST without variable Parameters should compile with an empty input vector."""
    compiled = compile_ast([Literal(2) * 3 + 1])

    assert compiled.variables == ()
    assert compiled.initial_values.size == 0

    result = compiled.evaluate(np.array([]))
    np.testing.assert_array_equal(result, [7.0])


def test_fixed_parameter_is_compiled_as_constant():
    """
    Fixed Parameters remain design values but do not occupy variable slots.

    Their current value is embedded into the compiled program.
    """
    x = Parameter(2, name="x")
    compiled = compile_ast([x * 3])

    assert compiled.variables == ()

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([])),
        [6.0],
    )


# ------------------
# VARIABLE IDENTITY
# ------------------
def test_shared_parameter_appears_once():
    """
    Reusing the same Parameter object must produce one variable slot,
    even when it appears in multiple roots.
    """
    x = Parameter(2, name="x").variable()

    compiled = compile_ast([x * 2, x * 3])

    assert compiled.variables == (x,)

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([5.0])),
        [10.0, 15.0],
    )


def test_distinct_parameters_with_same_metadata_remain_distinct():
    """
    Variable identity is based on object identity.

    Equal names and values must not merge distinct Parameter objects.
    """
    a = Parameter(1, name="x").variable()
    b = Parameter(1, name="x").variable()

    compiled = compile_ast([a + b])

    assert compiled.variables == (a, b)
    assert compiled.evaluate(np.array([2.0, 3.0]))[0] == 5.0


def test_wrong_number_of_variable_values():
    """Evaluation must reject vectors that do not match the compiled variable count."""
    x = Parameter(1, name="x").variable()
    y = Parameter(2, name="y").variable()

    compiled = compile_ast([x + y])

    with pytest.raises(ValueError):
        compiled.evaluate(np.array([1.0]))

    with pytest.raises(ValueError):
        compiled.evaluate(np.array([1.0, 2.0, 3.0]))


# -----------------------
# GRAPH SHARING / MEMOING
# -----------------------
def test_diamond_graph():
    """
    Shared dependencies should compile correctly when branches split and recombine.

           -> x * 2 -
        x             -> sum
           -> x * 3 -
    """
    x = Parameter(2, name="x").variable()

    left = x * 2
    right = x * 3
    root = left + right

    compiled = compile_ast([root])

    assert compiled.evaluate(np.array([5.0]))[0] == 25.0


def test_shared_subexpression():
    """
    Explicitly shared expression nodes should compile to one reusable program slot.
    """
    x = Parameter(2, name="x").variable()

    shared = x + 1
    root = shared * shared

    compiled = compile_ast([root])

    assert compiled.evaluate(np.array([4.0]))[0] == 25.0


# ------------------
# CONTEXT RESOLUTION
# ------------------
def test_scalar_system_context():
    """A SystemVar may resolve directly to a scalar context value."""
    n = SystemVar("ambient_n")

    compiled = compile_ast(
        [n * 2],
        {"ambient_n": 1.5},
    )

    assert compiled.variables == ()
    assert compiled.evaluate(np.array([]))[0] == 3.0


def test_context_can_inject_variable_parameter():
    """
    Context values may themselves be AST nodes.

    A variable Parameter injected through a SystemVar must participate in the
    compiled variable vector like any directly referenced Parameter.
    """
    n = SystemVar("ambient_n")
    ambient_n = Parameter(1.5, name="ambient_n").variable()

    compiled = compile_ast(
        [n * 2],
        {"ambient_n": ambient_n},
    )

    assert compiled.variables == (ambient_n,)
    assert compiled.evaluate(np.array([2.0]))[0] == 4.0


def test_same_named_system_vars_share_context_entry():
    """
    SystemVar resolution is name-based, not identity-based.

    Distinct SystemVar objects with the same name should resolve to the same
    supplied context entry.
    """
    a = SystemVar("ambient_n")
    b = SystemVar("ambient_n")

    compiled = compile_ast(
        [a + b],
        {"ambient_n": 1.5},
    )

    assert compiled.evaluate(np.array([]))[0] == 3.0


def test_missing_system_context():
    """Compilation should fail clearly when a required SystemVar is unresolved."""
    var = SystemVar("missing")

    with pytest.raises(UnresolvedSystemVarError):
        compile_ast([var])


# ----------------
# INVALID GRAPHS
# ----------------
def test_empty_input_node_cannot_compile():
    """An unresolved InputNode cannot be part of a compiled AST."""
    handle = InputNode(None)

    with pytest.raises(UnresolvedInputError):
        compile_ast([handle])


def test_cycle_cannot_compile():
    """
    Compilation must detect graph cycles explicitly rather than recursing forever.
    """
    handle = InputNode(None)
    handle.node = handle + 1

    with pytest.raises(ASTCycleError):
        compile_ast([handle])


# ------------------
# SNAPSHOT SEMANTICS
# ------------------
def test_compilation_snapshots_fixed_parameter_value():
    """
    Fixed Parameter values are captured during compilation.

    Mutating the declarative Parameter afterwards must not affect an already
    compiled evaluator.
    """
    x = Parameter(2, name="x")
    compiled = compile_ast([x * 3])

    x.value = 100

    assert compiled.evaluate(np.array([]))[0] == 6.0


def test_compilation_snapshots_input_node_target():
    """
    InputNode indirection is resolved during compilation.

    Hot-swapping the declarative handle afterwards must not alter the compiled
    graph structure.
    """
    handle = InputNode(Parameter(2, name="x"))
    compiled = compile_ast([handle * 3])

    handle.node = Parameter(100, name="x")

    assert compiled.evaluate(np.array([]))[0] == 6.0


def test_compilation_snapshots_scalar_context():
    """Scalar context values are captured when the AST is compiled."""
    context = {"ambient_n": 1.5}
    var = SystemVar("ambient_n")

    compiled = compile_ast([var * 2], context)

    context["ambient_n"] = 10.0

    assert compiled.evaluate(np.array([]))[0] == 3.0


def test_compilation_snapshots_variable_structure():
    """
    Variable membership is fixed at compilation time.

    Changing a source Parameter to fixed afterwards must not alter the
    variable layout of an already compiled AST.
    """
    x = Parameter(2, name="x").variable()
    compiled = compile_ast([x * 3])

    x.fixed()

    assert compiled.variables == (x,)
    assert compiled.evaluate(np.array([4.0]))[0] == 12.0


def test_initial_values_are_snapshotted():
    """Compiled initial values should not track later Parameter mutations."""
    x = Parameter(2, name="x").variable()
    compiled = compile_ast([x])

    x.value = 100

    np.testing.assert_array_equal(compiled.initial_values, [2.0])


def test_evaluation_does_not_mutate_parameters():
    """
    Evaluation is pure with respect to the declarative AST.

    Variable values are supplied externally and must never be written back to
    the source Parameter objects.
    """
    x = Parameter(10, name="x").variable()
    compiled = compile_ast([x * 2])

    result = compiled.evaluate(np.array([50.0]))

    assert result[0] == 100.0
    assert x.value == 10.0


# -----------------
# DIFFERENTIABILITY
# -----------------
def test_compiled_ast_is_differentiable():
    """
    The compiled evaluator must remain compatible with Autograd.

    This is a core contract of the graph compiler, since downstream optimization
    depends on derivatives with respect to the variable-value vector.
    """
    x = Parameter(2, name="x").variable()
    compiled = compile_ast([x ** 2 + 3 * x])

    derivative = grad(lambda values: compiled.evaluate(values)[0])
    result = derivative(anp.array([4.0]))

    np.testing.assert_allclose(result, [11.0])


# --------------------
# PROGRAM OPTIMIZATION
# --------------------
def test_constant_folding():
    """
    Expressions containing only constants should be evaluated at compile time.
    """
    x = Parameter(2, name="x").variable()
    compiled = compile_ast([(Literal(2) + 3) * x])

    # The constant subtree 2 + 3 should collapse to one constant value.
    const_values = [
        instruction.value
        for instruction in compiled.program.instructions
        if instruction.opcode is Opcode.CONST
    ]

    assert const_values == [5]
    assert compiled.evaluate(np.array([4.0]))[0] == 20.0


def test_common_subexpression_elimination():
    """
    Structurally equivalent compiled expressions should share one result value.

    This optimization happens during compilation, even though the declarative
    AST deliberately does not intern equivalent expressions.
    """
    x = Parameter(2, name="x").variable()

    a = x + 1
    b = x + 1

    compiled = compile_ast([a, b])

    assert a is not b
    assert compiled.program.root_indices[0] == compiled.program.root_indices[1]

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([5.0])),
        [6.0, 6.0],
    )


def test_common_subexpression_elimination_does_not_merge_distinct_variables():
    """
    CSE must never merge independent Parameter objects merely because their
    values or metadata happen to match.
    """
    a = Parameter(1, name="x").variable()
    b = Parameter(1, name="x").variable()

    compiled = compile_ast([a, b])

    assert compiled.variables == (a, b)
    assert compiled.program.root_indices[0] != compiled.program.root_indices[1]

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([10.0, 20.0])),
        [10.0, 20.0],
    )


def test_dead_code_removed_after_constant_folding():
    """
    Intermediate instructions made obsolete by constant folding should not
    remain in the optimized program.
    """
    compiled = compile_ast([(Literal(2) + 3) * 4])

    # The entire expression should reduce to one constant instruction.
    assert len(compiled.program.instructions) == 1
    assert compiled.program.instructions[0].opcode is Opcode.CONST
    assert compiled.program.instructions[0].value == 20

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([])),
        [20.0],
    )


def test_duplicate_roots_are_preserved():
    """
    Root deduplication may reuse one computed value, but output multiplicity
    and ordering must remain unchanged.
    """
    x = Parameter(2, name="x").variable()
    expr = x * 3

    compiled = compile_ast([expr, expr, expr])

    assert compiled.program.root_indices == (
        compiled.program.root_indices[0],
        compiled.program.root_indices[0],
        compiled.program.root_indices[0],
    )

    np.testing.assert_array_equal(
        compiled.evaluate(np.array([4.0])),
        [12.0, 12.0, 12.0],
    )


def test_optimized_program_remains_differentiable():
    """
    Constant folding and CSE must not interfere with Autograd differentiation.
    """
    x = Parameter(2, name="x").variable()

    # Both constant folding and CSE have opportunities here.
    a = x * (Literal(2) + 3)
    b = x * (Literal(2) + 3)
    compiled = compile_ast([a + b])

    derivative = grad(lambda values: compiled.evaluate(values)[0])
    result = derivative(anp.array([4.0]))

    np.testing.assert_allclose(result, [10.0])