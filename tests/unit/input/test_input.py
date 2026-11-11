"""Shared declarations and the symbolic-to-numerical input boundary."""

from dataclasses import dataclass, field, fields
from types import MappingProxyType

import numpy as np
import pytest

from diffractix import GaussianBeam, Node, OpticalElement, OpticalInput, Parameter, ParaxialState, Ray, RayBundle, Symbol, parameter
from diffractix.backend import grad, hessian, jit, xp
from diffractix.graph import ASTCycleError, InputNode, UnresolvedInputError
from diffractix.system import AMBIENT_N


@dataclass(eq=False, kw_only=True)
class FieldDeclarations:
    direct: Node = 2.0
    subclass: Parameter = 3.0
    explicit: float = parameter(4.0)
    optional: Node | None = None
    plain: float = 5.0
    phase: complex = 1.0 + 2.0j
    samples: np.ndarray = field(default_factory=lambda: np.array([1.0j, 2.0j]))
    enabled: bool = True


@dataclass(eq=False, kw_only=True)
class ExampleInput(FieldDeclarations, OpticalInput):
    pass


@dataclass(eq=False, kw_only=True)
class ExampleElement(FieldDeclarations, OpticalElement):
    @property
    def matrix(self):
        return ((1.0, 0.0), (0.0, 1.0))

    @property
    def element_length(self):
        return 0.0


@dataclass(eq=False, kw_only=True)
class ParentInput(OpticalInput):
    signal: Node = 2.0
    doubled: Node = field(init=False)

    def __post_init__(self):
        self.doubled = 2 * self.signal


@dataclass(eq=False, kw_only=True)
class DerivedInput(ParentInput):
    gain: Node = 3.0
    total: Node = field(init=False)
    summary: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.total = self.doubled * self.gain
        self.summary = self.total + 1

    @property
    def power(self):
        return self.total**2


def test_paraxial_state_requires_a_propagation_implementation_in_both_forms():
    assert ParaxialState.result_columns == ()
    for state_type in (ParaxialState, ParaxialState.numerical_type()):
        with pytest.raises(TypeError, match="abstract.*propagate"):
            state_type()


@pytest.mark.parametrize("object_type", [ExampleInput, ExampleElement])
def test_inputs_and_elements_share_wrapping_and_variable_selection(object_type):
    samples = np.array([1.0j, 2.0j], dtype="complex64")
    ordinary = np.float32(5.0)
    source = object_type(samples=samples, plain=ordinary)

    assert source.parameter_names == ("direct", "subclass", "explicit", "optional")
    for name, value in (("direct", 2.0), ("subclass", 3.0), ("explicit", 4.0)):
        handle = getattr(source, name)
        assert isinstance(handle, InputNode)
        assert isinstance(handle.node, Parameter)
        assert handle.node.owner is source
        assert handle.value == value
    assert source.optional.node is None
    assert source.plain is ordinary
    assert source.samples is samples
    assert source.phase == 1.0 + 2.0j
    assert source.enabled is True

    assert source.variable() is source
    assert all(getattr(source, name).is_variable for name in ("direct", "subclass", "explicit"))
    assert source.fixed() is source
    assert not any(handle.is_variable for handle in source.parameters)


@pytest.mark.parametrize("object_type", [ExampleInput, ExampleElement])
def test_reassignment_keeps_handles_and_existing_expressions_live(object_type):
    source = object_type()
    handle = source.direct
    expression = 3 * handle
    external = Parameter(7.0)

    source.direct = external

    assert source.direct is handle
    assert handle.node is external
    assert external.owner is None
    assert expression.value == 21.0
    source.direct = 8.0
    assert source.direct is handle
    assert expression.value == 24.0


@pytest.mark.parametrize("value", [True, 1.0j, np.array([1.0, 2.0])])
def test_graph_fields_reject_values_outside_the_existing_parameter_model(value):
    with pytest.raises(TypeError, match="Parameter"):
        ExampleInput(direct=value)


def test_numeric_uses_current_parameters_without_mutating_the_declaration():
    external = Parameter(2.0).variable()
    source = ExampleInput(direct=2 * external + 1)
    originals = {item.name: getattr(source, item.name) for item in fields(source)}
    expression = source.direct.node
    external.value = 5.0

    first = source.numeric()

    assert first.direct == 11.0
    assert first.subclass == 3.0
    assert first.explicit == 4.0
    assert first.optional is None
    assert first.samples is source.samples
    assert first.phase == source.phase
    assert first.plain is source.plain
    assert first.parameter_names == ()
    assert all(getattr(source, name) is value for name, value in originals.items())
    assert source.direct.node is expression
    assert external.value == 5.0 and external.is_variable


def test_numeric_bindings_override_defaults_without_mutating_bindings_or_symbols():
    key = ("input", 42, "offset")
    symbol = Symbol(key, default=2.0)
    source = ExampleInput(direct=3 * symbol + 1)
    bindings = {key: 0.0}

    assert source.numeric().direct == 7.0
    assert source.numeric(bindings).direct == 1.0
    assert bindings == {key: 0.0}
    assert symbol.default == 2.0
    symbol.set_default(4.0)
    assert source.numeric().direct == 13.0


def test_shared_parameters_and_symbol_keys_do_not_cache_values_between_conversions():
    shared = Parameter(0.25).variable()
    key = ("input", "offset")
    source = Ray(x=shared + Symbol(key), theta=2 * shared + Symbol(key), n=1.0)
    x_handle, theta_handle = source.x, source.theta

    first = source.numeric({key: 1.0})
    shared.fixed()
    shared.value = 0.5
    second = source.numeric({key: 2.0})

    assert (first.x, first.theta) == (1.25, 1.5)
    assert (second.x, second.theta) == (2.5, 3.0)
    assert first is not second
    assert type(first) is type(second) is source.numerical_type()
    assert source.x is x_handle and source.theta is theta_handle
    assert shared.value == 0.5 and not shared.is_variable


def test_failed_cyclic_conversion_can_be_repaired_without_changing_prior_results():
    source = ExampleInput()
    previous = source.numeric()
    handle = source.direct
    source.direct = handle + 1

    with pytest.raises(ASTCycleError):
        source.numeric()

    source.direct = 9.0
    repaired = source.numeric()
    assert source.direct is handle
    assert previous.direct == 2.0
    assert repaired.direct == 9.0
    assert type(repaired) is type(previous)


def test_numeric_requires_a_binding_for_a_symbol_without_a_default():
    source = ExampleInput(direct=Symbol("required_input", default=None))

    with pytest.raises(KeyError, match="required_input"):
        source.numeric()


@pytest.mark.parametrize("value_factory", [
    pytest.param(lambda: 1.25, id="python-scalar"),
    pytest.param(lambda: np.float32(1.25), id="numpy-scalar"),
    pytest.param(lambda: xp.array(1.25, dtype="float32"), id="backend-scalar"),
])
def test_numerical_classes_are_cached_and_never_wrap_assignments(value_factory):
    numerical_type = ExampleInput.numerical_type()
    assert numerical_type is ExampleInput.numerical_type()
    assert numerical_type is ExampleInput().numeric().__class__
    assert numerical_type is not DerivedInput.numerical_type()
    value = value_factory()
    numerical = numerical_type(direct=value)

    assert isinstance(numerical, ExampleInput)
    assert numerical.direct is value
    assert numerical.parameter_names == ()
    assert numerical.parameters == ()
    replacement = value_factory()
    numerical.direct = replacement
    numerical.optional = replacement
    assert numerical.direct is replacement
    assert numerical.optional is replacement
    assert numerical.parameter_names == ()
    converted_again = numerical.numeric()
    assert type(converted_again) is numerical_type
    assert converted_again is not numerical
    assert converted_again.direct is replacement


def test_inherited_fields_and_post_init_recompute_numerical_derived_fields():
    parent = ParentInput()
    source = DerivedInput(signal=Parameter(2.0).variable(), gain=4.0)
    source.signal.value = 5.0

    numerical = source.numeric()

    assert parent.parameter_names == ("signal", "doubled")
    assert source.parameter_names == ("signal", "doubled", "gain", "total")
    assert isinstance(numerical, DerivedInput)
    assert numerical.signal == 5.0
    assert numerical.doubled == 10.0
    assert numerical.total == 40.0
    assert numerical.summary == 41.0
    assert numerical.power == 1600.0
    assert not isinstance(numerical.doubled, Node)
    assert not isinstance(numerical.total, Node)
    assert numerical.parameter_names == ()
    assert isinstance(source.doubled, InputNode)
    assert isinstance(source.total, InputNode)


def test_numeric_does_not_evaluate_stale_init_false_fields():
    source = DerivedInput(signal=5.0, gain=4.0)
    stale = Symbol("stale_derived_cache")
    source.doubled = stale
    source.total = stale
    source.summary = -999.0

    numerical = source.numeric()

    assert numerical.doubled == 10.0
    assert numerical.total == 40.0
    assert numerical.summary == 41.0
    assert source.doubled.node is stale
    assert source.total.node is stale
    assert source.summary == -999.0


def test_numeric_supports_inputs_without_graph_fields_and_preserves_numerical_data():
    @dataclass(eq=False)
    class NumericalOnly(OpticalInput):
        value: float
        phase: complex
        samples: np.ndarray

    scalar = np.float32(1.25)
    samples = np.array([], dtype="complex64")
    source = NumericalOnly(scalar, 1.0j, samples)

    numerical = source.numeric()

    assert source.parameter_names == numerical.parameter_names == ()
    assert numerical is not source
    assert numerical.value is scalar
    assert numerical.phase == 1.0j
    assert numerical.samples is samples
    assert type(numerical) is source.numerical_type()


def test_optional_graph_fields_keep_none_and_can_be_assigned_later():
    source = ExampleInput()
    handle = source.optional
    assert source.numeric().optional is None
    assert source.optional.node is None

    source.optional = Parameter(6.0).variable()
    assert source.optional is handle
    assert source.numeric().optional == 6.0
    source.optional = None
    assert source.optional is handle
    assert source.numeric().optional is None


@pytest.mark.parametrize("input_type, arguments, expected", [
    (Ray, {"x": 1e-3, "theta": 0.02}, ("x", "theta", "n")),
    (RayBundle, {"x": np.array([1e-3]), "theta": np.array([0.02])}, ("n",)),
    (GaussianBeam, {"q": 1.0j, "wavelength": 1e-6}, ("wavelength", "n")),
])
def test_builtin_inputs_declare_only_scalar_parameters_and_bind_ambient(input_type, arguments, expected):
    source = input_type(**arguments)
    assert source.parameter_names == expected
    assert isinstance(source.n.node, Symbol)
    assert source.n.node.key == AMBIENT_N.key

    with pytest.raises(KeyError, match="ambient_n"):
        source.numeric()
    numerical = source.numeric({AMBIENT_N.key: 1.5})
    assert numerical.refractive_index == 1.5
    assert numerical.parameter_names == ()


def test_numeric_resolves_node_bindings_and_aliases_without_mutating_them():
    parameter_value = Parameter(2.0).variable()
    expression = 3 * parameter_value + 1
    source = ExampleInput(direct=Symbol("alias"), explicit=Symbol("expression"))
    bindings = MappingProxyType({"alias": Symbol("expression"), "expression": expression})

    first = source.numeric(bindings)
    parameter_value.value = 4.0
    second = source.numeric(bindings)

    assert (first.direct, first.explicit) == (7.0, 7.0)
    assert (second.direct, second.explicit) == (13.0, 13.0)
    assert bindings["expression"] is expression
    assert bindings["alias"].key == "expression"
    assert source.direct.node.key == "alias"
    assert parameter_value.is_variable


def test_missing_alias_binding_can_be_repaired_without_rebuilding_input():
    source = ExampleInput(direct=Symbol("alias"))
    handle = source.direct
    bindings = {"alias": Symbol("required")}

    with pytest.raises(KeyError, match="required"):
        source.numeric(bindings)

    bindings["required"] = 8.0
    assert source.numeric(bindings).direct == 8.0
    assert source.direct is handle
    assert source.direct.node.key == "alias"


def test_numeric_rejects_empty_dependencies_but_can_retry_after_repair():
    empty = InputNode(None)
    source = ExampleInput(direct=empty + 1)

    with pytest.raises(UnresolvedInputError, match="empty InputNode"):
        source.numeric()

    empty.node = Parameter(3.0)
    assert source.numeric().direct == 4.0


def test_numeric_preserves_backend_scalar_binding_identity_and_dtype():
    source = ExampleInput(direct=Symbol("signal"))
    signal = xp.array(1.25, dtype="float32")

    numerical = source.numeric({"signal": signal})

    assert numerical.direct is signal
    assert numerical.direct.dtype == np.dtype("float32")
    assert numerical.numerical_type() is type(numerical)
    assert isinstance(source.direct, InputNode)


def test_numeric_binding_path_retains_gradients_hessians_and_jit():
    signal = Symbol("signal")
    source = ExampleInput(direct=signal.sin() + signal**2)
    evaluate = lambda value: source.numeric({"signal": value}).direct
    point = 0.4

    for function, expected in (
        (evaluate, np.sin(point) + point**2),
        (grad(evaluate), np.cos(point) + 2 * point),
        (hessian(evaluate), 2 - np.sin(point)),
    ):
        np.testing.assert_allclose(function(point), expected, rtol=1e-12)
        np.testing.assert_allclose(jit(function)(point), expected, rtol=1e-12)
    assert source.direct.node.left.operand is signal


def test_builtin_default_symbols_and_constraints_are_independent_per_input():
    first = Ray(x=0.0, theta=0.0)
    second = Ray(x=0.0, theta=0.0)
    constraint = first.x >= 0

    first.n.node.set_default(1.5)
    first.require(constraint)

    assert first.n.node is not second.n.node
    assert first.numeric().n == 1.5
    with pytest.raises(KeyError, match="ambient_n"):
        second.numeric()
    assert len(first.constraints) == 1 and first.constraints[0] is constraint
    assert second.constraints == ()


@pytest.mark.parametrize("input_type, arguments, quantity, expected", [
    (Ray, {"x": 1e-3, "theta": 0.01, "n": 1.0}, "x", 2e-3),
    (RayBundle, {"x": np.array([1e-3]), "theta": np.array([0.01]), "n": 1.0}, "x", [2e-3]),
    (GaussianBeam, {"q": 1j, "wavelength": 1e-6, "n": 1.0}, "q", 0.1 + 1j),
])
def test_custom_input_subclasses_remain_numerical_after_inherited_propagation(
    input_type, arguments, quantity, expected,
):
    @dataclass(eq=False)
    class CustomInput(input_type):
        calibration: float = 1.25

        @property
        def calibrated_value(self):
            return self.calibration * getattr(self, quantity)

    declarative = CustomInput(**arguments)
    source = declarative.numeric()
    propagated = source.propagate(1.0, 0.1, 0.0, 1.0, n=1.33)

    assert type(propagated) is type(source) is CustomInput.numerical_type()
    assert isinstance(propagated, CustomInput)
    assert propagated.parameter_names == ()
    np.testing.assert_allclose(getattr(propagated, quantity), expected)
    np.testing.assert_allclose(propagated.calibrated_value, 1.25 * np.asarray(expected))
    assert propagated.refractive_index == 1.33
    assert source.refractive_index == 1.0
    propagated.n = 1.5
    assert propagated.n == 1.5 and not isinstance(propagated.n, Node)


def test_inputs_get_unique_default_labels_but_numerical_copies_do_not():
    from diffractix import Ray
    first, second = Ray(x=0, theta=0, n=1), Ray(x=0, theta=0, n=1)
    assert first.label != second.label and first.label.startswith("Ray")
    assert Ray(x=0, theta=0, n=1, label="probe").label == "probe"
    assert first.numeric().label == first.label
    assert Ray.numerical_type()(x=0.0, theta=0.0, n=1.0).label is None
