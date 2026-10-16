import numpy as np
import pytest
from autograd import grad, hessian, jacobian
import autograd.numpy as anp
from types import SimpleNamespace

from diffractix.graph.node import Literal, Parameter, Symbol, InputNode
from diffractix.graph.compile import CompiledAST, compile_ast, Opcode
from diffractix.graph.ops import Op
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
def test_shared_subexpression():
    """
    Explicitly shared expression nodes should compile to one reusable program slot.
    """
    x = Parameter(2, name="x").variable()

    shared = x + 1
    root = shared * shared

    compiled = compile_ast([root])

    additions = [
        instruction for instruction in compiled.program.instructions
        if instruction.opcode is Opcode.BINARY and instruction.op is Op.ADD
    ]
    assert len(additions) == 1
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


def test_missing_runtime_symbol_binding_fails_clearly():
    """Evaluation identifies the key of every missing runtime binding."""
    compiled = compile_ast([Symbol("beam_width")])

    with pytest.raises(
        KeyError,
        match="Missing runtime binding for Symbol key 'beam_width'",
    ):
        compiled.evaluate(np.array([]))


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
def test_parameter_snapshot_fixed_value_overrides_later_mutation():
    """A snapshotted fixed value is authoritative over live Parameter state."""
    parameter = Parameter(1.0, name="p")
    snapshot = {
        id(parameter): SimpleNamespace(value=1.0, parameter_index=None),
    }
    parameter.value = 5.0

    compiled = compile_ast(
        [parameter * 2],
        parameter_snapshot=snapshot,
    )

    assert compiled.variables == ()
    np.testing.assert_array_equal(compiled.evaluate(np.array([])), [2.0])


def test_parameter_snapshot_variable_membership_survives_live_fixed_change():
    """Snapshot variable membership wins if the live Parameter becomes fixed."""
    parameter = Parameter(1.0, name="p").variable()
    snapshot = {
        id(parameter): SimpleNamespace(value=1.0, parameter_index=0),
    }
    parameter.fixed()

    compiled = compile_ast(
        [parameter * 2],
        parameter_snapshot=snapshot,
    )

    assert len(compiled.variables) == 1
    assert compiled.variables[0] is parameter
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([3.0])),
        [6.0],
    )


def test_parameter_snapshot_fixed_membership_survives_live_variable_change():
    """Snapshot fixed membership wins if the live Parameter becomes variable."""
    parameter = Parameter(1.0, name="p")
    snapshot = {
        id(parameter): SimpleNamespace(value=1.0, parameter_index=None),
    }
    parameter.variable()

    compiled = compile_ast(
        [parameter * 2],
        parameter_snapshot=snapshot,
    )

    assert compiled.variables == ()
    np.testing.assert_array_equal(compiled.evaluate(np.array([])), [2.0])


def test_parameter_snapshot_value_populates_compiled_initial_values():
    """Variable initial values come from snapshot metadata rather than live state."""
    parameter = Parameter(1.0, name="p").variable()
    snapshot = {
        id(parameter): SimpleNamespace(value=1.0, parameter_index=0),
    }
    parameter.value = 5.0

    compiled = compile_ast(
        [parameter],
        parameter_snapshot=snapshot,
    )

    np.testing.assert_array_equal(compiled.initial_values, [1.0])


def test_parameter_absent_from_snapshot_uses_live_state():
    """A partial snapshot falls back to live state for unknown Parameters."""
    snapshotted = Parameter(1.0, name="snapshotted")
    live = Parameter(2.0, name="live")
    snapshot = {
        id(snapshotted): SimpleNamespace(value=1.0, parameter_index=None),
    }
    live.value = 4.0
    live.variable()

    compiled = compile_ast(
        [live * 2],
        parameter_snapshot=snapshot,
    )

    assert len(compiled.variables) == 1
    assert compiled.variables[0] is live
    np.testing.assert_array_equal(compiled.initial_values, [4.0])
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([3.0])),
        [6.0],
    )


def test_parameter_snapshot_applies_through_bound_symbol():
    """Symbol context indirection uses the same frozen Parameter metadata."""
    parameter = Parameter(2.0, name="gain").variable()
    snapshot = {
        id(parameter): SimpleNamespace(value=2.0, parameter_index=0),
    }
    parameter.value = 9.0
    parameter.fixed()

    compiled = compile_ast(
        [Symbol("gain") * 2],
        context={"gain": parameter},
        parameter_snapshot=snapshot,
    )

    assert len(compiled.variables) == 1
    assert compiled.variables[0] is parameter
    assert compiled.symbols == ()
    np.testing.assert_array_equal(compiled.initial_values, [2.0])
    np.testing.assert_array_equal(
        compiled.evaluate(np.array([3.0])),
        [6.0],
    )


def test_compilation_snapshots_fixed_parameter_value():
    """
    Fixed Parameter values are captured during compilation.

    Mutating the declarative Parameter afterwards must not affect an already
    compiled evaluator.
    """
    x = Parameter(2, name="x")
    compiled = compile_ast([x * 3])

    assert compiled.variables == ()
    assert compiled.symbols == ()
    assert compiled.initial_values.size == 0
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
    with pytest.raises(ValueError):
        compiled.initial_values[0] = 3.0


@pytest.mark.parametrize("variable", (False, True))
@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -np.inf, 2.0))
def test_compile_rejects_nonfinite_and_out_of_bounds_parameter_values(
    variable,
    invalid_value,
):
    parameter = Parameter(
        0.5,
        name="validated",
        variable=variable,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    parameter.value = invalid_value

    with pytest.raises(ValueError, match="validated|Parameter"):
        compile_ast([parameter])


@pytest.mark.parametrize(
    ("attribute", "value", "exception"),
    (
        ("lower_bound", np.nan, ValueError),
        ("upper_bound", True, TypeError),
        ("lower_bound", 2.0, ValueError),
        ("value", 2.0, ValueError),
    ),
)
def test_compile_defensively_validates_mutated_parameter_state(
    attribute,
    value,
    exception,
):
    parameter = Parameter(
        0.5,
        name="mutated",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    setattr(parameter, attribute, value)

    with pytest.raises(exception):
        compile_ast([parameter])


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

    assert compiled.variables == ()
    assert compiled.symbols == ()
    assert compiled.initial_values.size == 0
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


def test_hybrid_alias_dag_preserves_root_order_and_all_input_derivatives():
    x = Parameter(1.0).variable()
    y = Parameter(2.0).variable()
    offset = Parameter(1.0)
    snapshot = {id(offset): SimpleNamespace(value=1.0, parameter_index=None)}
    handle = InputNode(y)
    shared = (Literal(2) + 3) * handle + x
    runtime_key = ("result", 42, "gain")
    expression = shared**2 + Symbol("calibration") * x
    bindings = {"calibration": Symbol(runtime_key) + offset}
    compiled = compile_ast([expression, -expression, shared, expression], context=bindings, parameter_snapshot=snapshot)

    assert len(compiled.variables) == 2
    assert compiled.variables[0] is y
    assert compiled.variables[1] is x
    assert tuple(symbol.key for symbol in compiled.symbols) == (runtime_key,)

    # Change both declarative indirections after compilation, not just values.
    handle.node = Parameter(99.0).variable()
    offset.value = 100.0
    bindings["calibration"] = Literal(-100.0)

    def evaluate(inputs):
        return compiled.evaluate(inputs[:2], bindings={runtime_key: inputs[2]})

    inputs = anp.array([3.0, 4.0, 2.0])  # [y, x, runtime_gain]
    np.testing.assert_allclose(evaluate(inputs), [373.0, -373.0, 19.0, 373.0])
    scalar = lambda values: evaluate(values)[0]
    np.testing.assert_allclose(grad(scalar)(inputs), [190.0, 41.0, 4.0])
    np.testing.assert_allclose(hessian(scalar)(inputs), [[50.0, 10.0, 0.0], [10.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(evaluate(anp.array([3.0, 4.0, 5.0])), [385.0, -385.0, 19.0, 385.0])


def test_recursive_symbol_alias_cycle_does_not_poison_repaired_compilation():
    parameter = Parameter(2.0).variable()
    root = Symbol("outer") + Symbol("outer")
    context = {"outer": Symbol("inner"), "inner": parameter}
    original = compile_ast([root], context=context)

    # Distinct Symbol objects with the same keys must not evade cycle detection.
    context["inner"] = Symbol("outer") + 1
    with pytest.raises(ASTCycleError):
        compile_ast([root], context=context)

    context["inner"] = 3 * parameter
    repaired = compile_ast([root], context=context)
    np.testing.assert_allclose(original.evaluate(np.array([4.0])), [8.0])
    np.testing.assert_allclose(repaired.evaluate(np.array([4.0])), [24.0])
    assert original.symbols == repaired.symbols == ()


@pytest.mark.parametrize("left,right,dtype", [
    (1.0, 2.0, "float64"),
    (np.float32(1), np.float32(2), "float32"),
    (np.float32(1), 2.0, "float64"),
    (np.int16(1), np.float32(2), "float32"),
    (1, 2, np.dtype(int)),
    (True, False, "bool"),
    (1.0, 2j, "complex128"),
    (np.complex64(1j), np.float32(2), "complex64"),
    (2 ** 80, 1.0, "object"),
    (np.array(1.0), np.array(2.0), "float64"),
    (np.array([1, 2], dtype="float32"), np.array([2j, 3j], dtype="complex64"), "complex64"),
])
def test_sequence_outputs_preserve_packed_root_order_shape_and_dtype(left, right, dtype):
    a, b = Symbol("a"), Symbol("b")
    graph = compile_ast((a, b, a + b, a))
    bindings = {"a": left, "b": right}

    packed = graph.evaluate(np.array([]), bindings=bindings)
    sequence = graph._evaluate_sequence(np.array([]), bindings=bindings)

    assert isinstance(packed, np.ndarray)
    assert isinstance(sequence, tuple)
    assert packed.dtype == np.dtype(dtype)
    assert np.array(sequence).dtype == packed.dtype
    assert np.array(sequence).shape == packed.shape
    np.testing.assert_array_equal(sequence, packed)
    np.testing.assert_array_equal(graph(np.array([]), bindings), packed)


def test_scalar_float64_sequence_does_not_pack_an_array(monkeypatch):
    parameter = Parameter(0.5).variable()
    graph = compile_ast((Literal(1.0), parameter, parameter + Symbol("offset"), parameter))
    theta = np.array([0.75])

    def forbidden_array(*args, **kwargs):
        raise AssertionError("Scalar sequence evaluation must not pack roots")

    monkeypatch.setattr("diffractix.graph.compile.np.array", forbidden_array)
    sequence = graph._evaluate_sequence(theta, bindings={"offset": 0.25})

    assert sequence == (1.0, 0.75, 1.0, 0.75)
    assert all(isinstance(value, np.float64) for value in sequence)


@pytest.mark.parametrize("binding", [np.array(1.5), np.array([1.5, 2.0])])
def test_sequence_array_bindings_preserve_packed_copy_semantics(binding):
    graph = compile_ast((Symbol("n"),))
    sequence = graph._evaluate_sequence(np.array([]), bindings={"n": binding})
    expected = np.array(sequence, copy=True)

    binding[...] = 9.0

    np.testing.assert_array_equal(sequence, expected)


@pytest.mark.parametrize("method", ["evaluate", "_evaluate_sequence"])
def test_sequence_and_public_evaluation_share_validation(method):
    parameter = Parameter(1.0).variable()
    graph = compile_ast((parameter, parameter + Symbol("missing")))
    evaluate = getattr(graph, method)

    with pytest.raises(ValueError, match="Expected 1 variable values"):
        evaluate(np.array([]))
    with pytest.raises(KeyError, match="missing"):
        evaluate(np.array([1.0]))
    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        evaluate(np.array([1.0]), bindings={"missing": np.array([1.0, 2.0])})


def test_empty_graph_sequence_and_public_return_types_are_unchanged():
    graph = compile_ast(())
    assert graph._evaluate_sequence(np.array([])) == ()
    packed = graph.evaluate(np.array([]))
    assert isinstance(packed, np.ndarray)
    assert packed.shape == (0,)
    assert packed.dtype == np.dtype(float)


@pytest.mark.parametrize("kind", ["real", "complex", "array"])
def test_sequence_outputs_match_packed_nested_and_fourth_order_derivatives(kind):
    parameter = Parameter(0.4).variable()
    root = parameter ** 3 + Symbol("coefficient") * parameter ** 2
    graph = compile_ast((root, root + 1, root ** 2))
    theta = np.array([0.4, 0.2, 0.7])

    def values(t, sequence):
        coefficient = {
            "real": lambda: t[1] ** 2 + t[2] ** 3,
            "complex": lambda: t[1] ** 2 + 1j * t[2],
            "array": lambda: t[1:] ** 2,
        }[kind]()
        evaluate = graph._evaluate_sequence if sequence else graph.evaluate
        result = evaluate(t[:1], bindings={"coefficient": coefficient})
        return anp.stack(result) if sequence else result

    def cost(t, sequence):
        result = values(t, sequence)
        return anp.sum(anp.real(result * anp.conj(result)))

    reference = lambda t: cost(t, False)
    optimized = lambda t: cost(t, True)
    np.testing.assert_allclose(values(theta, True), values(theta, False), rtol=1e-12)

    def real_values(t, sequence):
        result = values(t, sequence)
        return anp.stack((anp.real(result), anp.imag(result)))

    np.testing.assert_allclose(jacobian(lambda t: real_values(t, True))(theta),
                               jacobian(lambda t: real_values(t, False))(theta), rtol=1e-10, atol=1e-12)
    for derivative in (grad, hessian, lambda f: jacobian(hessian(f)),
                       lambda f: jacobian(jacobian(hessian(f)))):
        np.testing.assert_allclose(derivative(optimized)(theta), derivative(reference)(theta),
                                   rtol=1e-9, atol=1e-10, equal_nan=False)

    # Differentiate a function that itself differentiates through the evaluator.
    nested = lambda t, f: anp.sum(grad(f)(t) ** 2)
    np.testing.assert_allclose(hessian(lambda t: nested(t, optimized))(theta),
                               hessian(lambda t: nested(t, reference))(theta),
                               rtol=1e-9, atol=1e-10, equal_nan=False)
