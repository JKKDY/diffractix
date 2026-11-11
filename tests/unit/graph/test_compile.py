"""Program compilation preserves graph meaning and freezes declarative structure."""
import numpy as np
import pytest

from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.graph import (
    ASTCycleError, InputNode, Literal, Node, Parameter, Program, Symbol,
    UnresolvedInputError, UnsupportedNodeError, compile_graph, precompile,
)


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_compilation_preserves_root_order_and_independent_parameter_identity(compiler):
    first = Parameter(1.0, name="same").variable()
    second = Parameter(1.0, name="same").variable()
    program = compiler((second + first, second, first, second + first))
    assert isinstance(program, Program)
    assert program.parameter_indices == {id(second): 0, id(first): 1}
    assert program.runtime_symbols == {}
    result = program.evaluate([2.0, 5.0])
    assert isinstance(result, tuple)
    np.testing.assert_array_equal(result, [7.0, 2.0, 5.0, 7.0])
    assert first.value == second.value == 1.0
    assert first.is_variable and second.is_variable


@pytest.mark.parametrize("binding_factory,variable", [
    (lambda p: 2.0, False), (lambda p: Literal(2.0), False),
    (lambda p: p, False), (lambda p: p, True),
    (lambda p: InputNode(3 * p + 1), True),
])
def test_compile_time_binding_resolves_as_graph_content(binding_factory, variable):
    parameter = Parameter(2.0, variable=variable)
    binding = binding_factory(parameter)
    program = compile_graph((Symbol("external") * 2,), {"external": binding})
    assert program.runtime_symbols == {}
    assert program.parameter_indices == ({id(parameter): 0} if variable else {})
    expected = 20.0 if isinstance(binding, InputNode) else (6.0 if variable else 4.0)
    assert program.evaluate([3.0] if variable else [])[0] == expected


def test_symbol_aliases_can_end_at_a_runtime_symbol_and_share_one_slot():
    parameter = Parameter(1.0).variable()
    key = ("result", 42, "width")
    program = compile_graph(
        (parameter + Symbol("alias") + Symbol(key), Symbol("other")),
        {"alias": Symbol(key)},
    )
    assert program.parameter_indices == {id(parameter): 0}
    assert program.runtime_symbols == {key: 0, "other": 1}
    np.testing.assert_allclose(program.evaluate([2.0], {"other": 7.0, key: 3.0}), [8.0, 7.0])


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_compilation_snapshots_handle_expression_bindings_and_variable_membership(compiler):
    variable = Parameter(2.0).variable()
    fixed = Parameter(3.0)
    handle = InputNode(variable)
    expression = handle * fixed
    bindings = {"expression": expression}
    program = compiler((Symbol("expression"),), bindings)
    variable.fixed()
    fixed.value = 100.0
    handle.node = Parameter(100.0).variable()
    expression.right = Literal(500.0)
    bindings["expression"] = Literal(-100.0)
    assert program.parameter_indices == {id(variable): 0}
    assert program.evaluate([4.0])[0] == 12.0


@pytest.mark.parametrize("symbol", [Symbol("missing"), Symbol("missing", default=2.0)])
def test_runtime_symbols_do_not_use_interpreter_defaults(symbol):
    program = compile_graph((symbol,))
    assert program.runtime_symbols == {"missing": 0}
    with pytest.raises(KeyError, match="missing"):
        program.evaluate([])
    assert program.evaluate([], {"missing": 0.0, "unused": 99.0}) == (0.0,)


@pytest.mark.parametrize("theta", [[], [1.0, 2.0]])
def test_evaluation_rejects_wrong_theta_length(theta):
    program = compile_graph((Parameter(1.0).variable(),))
    with pytest.raises(ValueError, match="Expected 1 variable values"):
        program.evaluate(theta)


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_empty_and_constant_programs_do_not_require_variables(compiler):
    empty = compiler(())
    assert empty.evaluate([]) == ()
    assert empty.parameter_indices == empty.runtime_symbols == {}
    assert compiler((Literal(2), Literal(-0.0), Literal(1j))).evaluate([]) == (2, -0.0, 1j)


def test_native_roots_keep_mixed_shapes_and_dtypes_without_common_packing():
    program = compile_graph((Symbol("real"), Symbol("complex"), Symbol("array"), Symbol("real")))
    bindings = {
        "real": np.float32(1.25),
        "complex": np.complex128(2 + 3j),
        "array": xp.array([2.0, 4.0], dtype="float32"),
    }
    result = program.evaluate([], bindings)
    assert isinstance(result, tuple)
    assert result[0] is result[3] is bindings["real"]
    assert result[0].dtype == np.dtype("float32")
    assert result[1].dtype == np.dtype("complex128")
    np.testing.assert_array_equal(result[2], [2.0, 4.0])
    assert result[2].shape == (2,)


@pytest.mark.parametrize("root,error", [
    (1, TypeError), (InputNode(None), UnresolvedInputError), (Node(), UnsupportedNodeError),
])
def test_compilation_rejects_invalid_roots(root, error):
    with pytest.raises(error):
        compile_graph((root,))


def test_recursive_symbol_alias_cycle_does_not_poison_repaired_compilation():
    parameter = Parameter(2.0).variable()
    root = Symbol("outer") + Symbol("outer")
    bindings = {"outer": Symbol("inner"), "inner": parameter}
    original = compile_graph((root,), bindings)
    bindings["inner"] = Symbol("outer") + 1
    with pytest.raises(ASTCycleError):
        compile_graph((root,), bindings)
    bindings["inner"] = 3 * parameter
    repaired = compile_graph((root,), bindings)
    assert original.evaluate([4.0]) == (8.0,)
    assert repaired.evaluate([4.0]) == (24.0,)
    assert original.runtime_symbols == repaired.runtime_symbols == {}


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf, 2.0])
def test_compilation_defensively_rejects_mutated_bounded_parameter_values(compiler, variable, invalid_value):
    parameter = Parameter(0.5, name="validated", variable=variable, bounds=(0.0, 1.0))
    parameter.value = invalid_value
    with pytest.raises(ValueError, match="validated|parameter|Parameter"):
        compiler((parameter * 2,))


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("field,value,error", [
    ("lower_bound", True, TypeError), ("upper_bound", "bad", TypeError),
    ("lower_bound", np.nan, ValueError), ("upper_bound", np.nan, ValueError),
    ("lower_bound", 2.0, ValueError), ("upper_bound", -1.0, ValueError),
])
def test_compilation_defensively_rejects_mutated_parameter_bounds(variable, field, value, error):
    parameter = Parameter(0.5, name="validated", variable=variable, bounds=(0.0, 1.0))
    setattr(parameter, field, value)
    with pytest.raises(error):
        compile_graph((parameter,))


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("value,error", [
    (np.nan, ValueError), (True, TypeError), (1j, TypeError), ("1.0", TypeError),
])
def test_compilation_rejects_invalid_mutated_values_on_unbounded_parameters(variable, value, error):
    parameter = Parameter(0.5, name="unbounded", variable=variable)
    parameter.value = value
    with pytest.raises(error, match="unbounded"):
        compile_graph((parameter * 2,))


def test_infinite_fixed_optical_radius_remains_a_valid_constant():
    radius = Parameter(np.inf)
    assert compile_graph((-2.0 / radius,)).evaluate([])[0] == 0.0


def test_hybrid_alias_dag_preserves_root_order_snapshots_and_all_input_derivatives():
    x, y = Parameter(1.0).variable(), Parameter(2.0).variable()
    offset = Parameter(1.0)
    handle = InputNode(y)
    shared = (Literal(2) + 3) * handle + x
    key = ("result", 42, "gain")
    expression = shared**2 + Symbol("calibration") * x
    bindings = {"calibration": Symbol(key) + offset}
    program = compile_graph((expression, -expression, shared, expression), bindings)
    assert program.parameter_indices == {id(y): 0, id(x): 1}
    handle.node = Parameter(99.0).variable()
    offset.value = 100.0
    bindings["calibration"] = Literal(-100.0)
    def evaluate(point):
        return xp.stack(program.evaluate(point[:2], {key: point[2]}))
    point = xp.array([3.0, 4.0, 2.0])
    np.testing.assert_allclose(evaluate(point), [373.0, -373.0, 19.0, 373.0])
    scalar = lambda values: evaluate(values)[0]
    for function in (grad(scalar), jit(grad(scalar))):
        np.testing.assert_allclose(function(point), [190.0, 41.0, 4.0])
    for function in (hessian(scalar), jit(hessian(scalar))):
        np.testing.assert_allclose(function(point), [[50.0, 10.0, 0.0], [10.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(evaluate(xp.array([3.0, 4.0, 5.0])), [385.0, -385.0, 19.0, 385.0])


@pytest.mark.parametrize("kind", ["real", "complex", "array"])
@pytest.mark.parametrize("derivative", ["gradient", "hessian", "third", "nested"])
def test_program_and_analytical_expression_agree_through_nested_derivatives(kind, derivative):
    parameter = Parameter(0.4).variable()
    root = parameter**3 + Symbol("coefficient") * parameter**2
    program = compile_graph((root, root + 1, root**2))
    def coefficient(t):
        if kind == "real":
            return t[1]**2 + t[2]**3
        if kind == "complex":
            return t[1]**2 + 1j*t[2]
        return t[1:]**2
    def actual(t):
        return xp.stack(program.evaluate(t[:1], {"coefficient": coefficient(t)}))
    def expected(t):
        value = t[0]**3 + coefficient(t)*t[0]**2
        return xp.stack((value, value + 1, value**2))
    def cost(function, t):
        values = function(t)
        return xp.sum(xp.real(values * xp.conj(values)))
    point = xp.array([0.4, 0.2, 0.7])
    np.testing.assert_allclose(actual(point), expected(point), rtol=1e-12)
    def nested(function):
        return hessian(lambda t: xp.sum(grad(function)(t)**2))
    transform = {
        "gradient": grad,
        "hessian": hessian,
        "third": lambda f: jacobian(hessian(f)),
        "nested": nested,
    }[derivative]
    def analytical_cost(t):
        value = t[0]**3 + coefficient(t)*t[0]**2
        real, imaginary = xp.real(value), xp.imag(value)
        magnitude_squared = real**2 + imaginary**2
        # |r|**2 + |r+1|**2 + |r**2|**2, without stacked-root adjoints.
        return xp.sum(magnitude_squared**2 + 2*magnitude_squared + 2*real + 1)
    reference = transform(analytical_cost)(point)
    np.testing.assert_allclose(
        transform(lambda t: cost(actual, t))(point), reference,
        rtol=1e-9, atol=1e-10, equal_nan=False,
    )


def test_program_text_lists_inputs_instructions_and_outputs_in_order():
    """str(program) is the debugging view: theta slots, runtime symbols, code, outputs."""
    from diffractix.graph import Literal, Parameter, Symbol
    parameter = Parameter(0.5).variable()
    program = compile_graph(((parameter * 2 + Symbol("k")).sin(), parameter, Literal(3.0)))
    lines = str(program).splitlines()

    assert lines[0] == "INPUTS"
    assert lines[1].split("\t")[:2] == ["  i0", "theta[0]"] and str(id(parameter)) in lines[1]
    assert lines[2].split("\t") == ["  i1", "symbol['k']"]
    code = lines[lines.index("PROGRAM") + 1:lines.index("OUTPUTS") - 1]
    assert [line.split()[0] for line in code] == ["MUL", "ADD", "SIN"]
    assert code[0].split()[1:] == ["r0,", "i0,", "2"]
    assert [line.split("\t")[1] for line in lines[lines.index("OUTPUTS") + 1:]] == ["r0", "i0", "3.0"]


def test_program_text_of_an_empty_program_has_no_instructions():
    text = str(compile_graph(()))
    assert "INPUTS" in text and "OUTPUTS" in text
    assert text.split("PROGRAM")[1].split("OUTPUTS")[0].strip() == ""


@pytest.mark.parametrize("compiler", [precompile, compile_graph])
def test_compilation_rejects_empty_handles_and_unknown_node_types(compiler):
    from diffractix.graph import InputNode, Node, UnresolvedInputError, UnsupportedNodeError

    class Unknown(Node):
        pass

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        compiler((InputNode(None) + 1,))
    with pytest.raises(UnsupportedNodeError, match="Unknown"):
        compiler((Unknown(),))
