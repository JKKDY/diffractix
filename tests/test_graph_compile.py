import numpy as np
import pytest
from autograd import grad
import autograd.numpy as anp

from diffractix.graph.node import Literal, Parameter, Symbol, InputNode
from diffractix.graph.compile import CompiledAST, compile_ast, Opcode
from diffractix.graph.utils import (
    ASTCycleError,
    UnresolvedInputError,
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
    assert compiled.symbols == ()
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


def test_complex_expression_compiles():
    x = Parameter(2.0).variable()
    compiled = compile_ast([-1j * x])

    result = compiled.evaluate(np.array([3.0]))

    np.testing.assert_allclose(result, [-3j])


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
def test_compile_time_symbol_binding_to_scalar():
    """A Symbol may resolve directly to a scalar context value."""
    n = Symbol("ambient_n")

    compiled = compile_ast(
        [n * 2],
        {"ambient_n": 1.5},
    )

    assert compiled.variables == ()
    assert compiled.symbols == ()
    assert compiled.evaluate(np.array([]))[0] == 3.0


def test_compile_time_symbol_binding_to_literal():
    """A Literal binding is folded and does not become theta or runtime input."""
    symbol = Symbol("offset")

    compiled = compile_ast(
        [symbol + 1],
        {"offset": Literal(2.5)},
    )

    assert compiled.variables == ()
    assert compiled.symbols == ()
    np.testing.assert_array_equal(compiled.evaluate(np.array([])), [3.5])


def test_compile_time_symbol_binding_to_fixed_parameter():
    """A fixed Parameter bound through context is compiled as a constant."""
    symbol = Symbol("focal_length")
    focal_length = Parameter(2.0, name="focal_length")

    compiled = compile_ast(
        [symbol * 3],
        {"focal_length": focal_length},
    )

    assert compiled.variables == ()
    assert compiled.symbols == ()
    assert compiled.evaluate(np.array([]))[0] == 6.0


def test_compile_time_symbol_binding_to_variable_parameter():
    """
    Context values may themselves be AST nodes.

    A variable Parameter injected through a Symbol must participate in the
    compiled variable vector like any directly referenced Parameter.
    """
    n = Symbol("ambient_n")
    ambient_n = Parameter(1.5, name="ambient_n").variable()

    compiled = compile_ast(
        [n * 2],
        {"ambient_n": ambient_n},
    )

    assert compiled.variables == (ambient_n,)
    assert compiled.symbols == ()
    assert compiled.evaluate(np.array([2.0]))[0] == 4.0


def test_compile_time_symbol_binding_to_expression_discovers_theta():
    """Recursive bindings expose only variable Parameters from an expression."""
    scale = Parameter(2.0, name="scale").variable()
    offset = Parameter(3.0, name="offset")
    symbol = Symbol("calibration")

    compiled = compile_ast(
        [2 * symbol],
        {"calibration": 3 * scale + offset},
    )

    assert compiled.variables == (scale,)
    assert compiled.symbols == ()
    np.testing.assert_array_equal(compiled.initial_values, [2.0])
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([4.0])),
        [30.0],
    )


def test_parameter_variability_controls_theta_for_bound_symbol():
    """Recompilation reflects whether a Symbol's Parameter binding is variable."""
    parameter = Parameter(2.0, name="gain")
    symbol = Symbol("gain")
    context = {"gain": parameter}

    fixed_compiled = compile_ast([symbol], context)
    parameter.variable()
    variable_compiled = compile_ast([symbol], context)

    assert fixed_compiled.variables == ()
    assert fixed_compiled.symbols == ()
    np.testing.assert_array_equal(
        fixed_compiled.evaluate(np.array([])),
        [2.0],
    )

    assert variable_compiled.variables == (parameter,)
    assert variable_compiled.symbols == ()
    np.testing.assert_array_equal(variable_compiled.initial_values, [2.0])
    np.testing.assert_array_equal(
        variable_compiled.evaluate(np.array([5.0])),
        [5.0],
    )


def test_compile_time_symbol_bindings_resolve_recursively():
    """A chain of bound Symbols resolves fully to ordinary graph content."""
    parameter = Parameter(1.5, name="ambient_n").variable()
    outer = Symbol("outer")

    compiled = compile_ast(
        [outer * 2],
        {
            "outer": Symbol("inner"),
            "inner": parameter,
        },
    )

    assert compiled.variables == (parameter,)
    assert compiled.symbols == ()
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([2.0])),
        [4.0],
    )


def test_bound_symbol_can_resolve_to_unbound_runtime_symbol():
    """Only the unresolved end of a Symbol chain becomes a runtime input."""
    outer = Symbol("outer")
    runtime_key = ("result", 42, "w")

    compiled = compile_ast(
        [outer + 1],
        {"outer": Symbol(runtime_key)},
    )

    assert compiled.variables == ()
    assert tuple(symbol.key for symbol in compiled.symbols) == (runtime_key,)
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([]), bindings={runtime_key: 3.0}),
        [4.0],
    )


def test_symbols_with_same_key_share_context_entry():
    """
    Symbol resolution is key-based, not identity-based.

    Distinct Symbol objects with the same key should resolve to the same
    supplied context entry.
    """
    a = Symbol("ambient_n")
    b = Symbol("ambient_n")

    compiled = compile_ast(
        [a + b],
        {"ambient_n": 1.5},
    )

    assert compiled.evaluate(np.array([]))[0] == 3.0


def test_unresolved_symbol_is_a_runtime_input():
    """An unbound Symbol should be exposed and evaluated from runtime bindings."""
    symbol = Symbol("scale")
    compiled = compile_ast([symbol * 3])

    assert len(compiled.symbols) == 1
    assert compiled.symbols[0] is symbol
    assert compiled.evaluate(np.array([]), bindings={"scale": 2.0})[0] == 6.0


def test_theta_and_runtime_symbol_inputs_are_independent():
    """A graph may consume optimizer values and runtime bindings together."""
    x = Parameter(2.0, name="x").variable()
    scale = Symbol("scale")

    compiled = compile_ast([x * scale + 1])

    assert compiled.variables == (x,)
    assert tuple(symbol.key for symbol in compiled.symbols) == ("scale",)
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([4.0]), bindings={"scale": 3.0}),
        [13.0],
    )


def test_unresolved_symbols_preserve_first_occurrence_order():
    """Runtime Symbol indices follow deterministic left-to-right discovery."""
    x = Parameter(1.0, name="x").variable()
    a = Symbol("a")
    b = Symbol("b")

    compiled = compile_ast([x + a * b])

    assert tuple(symbol.key for symbol in compiled.symbols) == ("a", "b")
    assert compiled.evaluate(
        np.array([1.0]),
        bindings={"a": 2.0, "b": 3.0},
    )[0] == 7.0


def test_symbols_with_same_key_are_deduplicated():
    """Distinct Symbol objects with one key share one runtime input slot."""
    first = Symbol("shared")
    second = Symbol("shared")

    compiled = compile_ast([first + second])

    assert len(compiled.symbols) == 1
    assert compiled.symbols[0] is first
    assert compiled.evaluate(
        np.array([]),
        bindings={"shared": 2.5},
    )[0] == 5.0


def test_symbol_supports_tuple_key():
    """Structured hashable keys can identify runtime optical result values."""
    element = object()
    key = ("result", id(element), "w")
    symbol = Symbol(key)

    compiled = compile_ast([symbol + 1])

    assert tuple(item.key for item in compiled.symbols) == (key,)
    assert compiled.evaluate(np.array([]), bindings={key: 4.0})[0] == 5.0


def test_missing_runtime_symbol_binding_fails_clearly():
    """Evaluation identifies the key of every missing runtime binding."""
    compiled = compile_ast([Symbol("beam_width")])

    with pytest.raises(
        KeyError,
        match="Missing runtime binding for Symbol key 'beam_width'",
    ):
        compiled.evaluate(np.array([]))


def test_ast_without_symbols_needs_no_runtime_bindings():
    """Existing compiled AST evaluation remains unchanged without Symbols."""
    x = Parameter(2.0).variable()
    compiled = compile_ast([x + 1])

    assert compiled.symbols == ()
    np.testing.assert_array_equal(compiled.evaluate(np.array([4.0])), [5.0])


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
    var = Symbol("ambient_n")

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
