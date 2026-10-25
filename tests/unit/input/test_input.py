"""Shared declarations and the symbolic-to-numerical input boundary."""

from dataclasses import dataclass, field, fields

import numpy as np
import pytest

from diffractix import GaussianBeam, Node, OpticalElement, OpticalInput, Parameter, Ray, RayBundle, Symbol, parameter
from diffractix.backend import xp
from diffractix.graph import InputNode
from diffractix.graph.utils import ASTCycleError
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


@pytest.mark.parametrize("bindings", [None, {}])
def test_numeric_requires_a_binding_for_a_symbol_without_a_default(bindings):
    source = ExampleInput(direct=Symbol("required_input", default=None))

    with pytest.raises(KeyError, match="required_input"):
        source.numeric(bindings)


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
