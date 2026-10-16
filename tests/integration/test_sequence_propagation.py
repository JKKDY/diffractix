"""Sequence propagation against the original packed-array execution path."""

from copy import copy
from dataclasses import dataclass, fields
from types import SimpleNamespace

import autograd.numpy as np
from autograd import grad, hessian, jacobian
import numpy as numpy
import pytest

from diffractix.beams import GaussianBeam, ParaxialRay, RayBundle
from diffractix.beams.base import ParaxialState
from diffractix.composites import Slab, ThickLens
from diffractix.elements import ABCD, GaussianAperture, GRIN, Interface, OpticalElement, Space, ThinLens
from diffractix.graph import Node, Parameter, Symbol
from diffractix.solver import Solver, Constraint, Objective
from diffractix.solver.context import SolverContext
from diffractix.system import System
from tests.helpers.solver import packed_expression_reference


@dataclass(eq=False, kw_only=True)
class CustomElement(OpticalElement):
    shift: Node
    power: Node

    @property
    def matrix(self):
        return ((1.0, self.shift), (self.power, 1.0))

    @property
    def element_length(self):
        return self.shift


@dataclass
class CustomState(ParaxialState):
    x: object
    theta: object
    n: object = 1.0

    def propagate(self, A, B, C, D, n):
        return CustomState(A * self.x + B * self.theta, C * self.x + D * self.theta, n)


def packed_reference(simulation):
    """Same run()/metadata/source, using evaluate() exactly as before optimization."""
    reference = copy(simulation)
    reference.graph = SimpleNamespace(
        initial_values=simulation.initial_values,
        _evaluate_sequence=simulation.graph.evaluate,
    )
    return reference


def source_for(kind):
    if kind == "gaussian":
        return GaussianBeam.from_waist(w0=0.5e-3, wavelength=1064e-9)
    if kind == "ray":
        return ParaxialRay(x=1e-3, theta=0.01)
    if kind == "custom":
        return CustomState(x=np.array([1e-3, -2e-3]), theta=np.array([0.01, -0.02]))
    dtype = numpy.float32 if kind == "bundle32" else numpy.float64
    return RayBundle(x=np.array([1e-3, -2e-3], dtype=dtype),
                     theta=np.array([0.01, -0.02], dtype=dtype))


def observables(result):
    values = [result.z[-1]]
    values.extend(getattr(result.final, field.name) for field in fields(result.final))
    return np.concatenate([
        part
        for value in values
        for part in (np.reshape(np.real(value), (-1,)), np.reshape(np.imag(value), (-1,)))
    ])


def assert_finite_match(actual, expected, **kwargs):
    assert numpy.all(numpy.isfinite(actual))
    assert numpy.all(numpy.isfinite(expected))
    numpy.testing.assert_allclose(actual, expected, equal_nan=False, **kwargs)


@pytest.mark.parametrize("kind", ["gaussian", "ray", "bundle", "bundle32", "custom"])
@pytest.mark.parametrize("complex_matrix", [False, True])
@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part:numpy.exceptions.ComplexWarning")
def test_sequence_propagates_all_states_composites_and_custom_components(kind, complex_matrix):
    d = Parameter(0.025).variable(bounds=(0.01, 0.06))
    f = Parameter(0.15).variable(bounds=(0.08, 0.25))
    index = Parameter(1.4)
    repeated = Space(d=d)
    custom = CustomElement(shift=d / 3, power=(-0.001j if complex_matrix else -0.001) / f)
    simulation = System(
        repeated, custom,
        Slab(d=0.004 + 0.1 * d, n=index),
        ThickLens(d=0.003, n=index, R1=0.1, R2=-0.1),
        Interface(n1=1.0, n2=index, R=0.1),
        Space(d=d / 2, n=index),
        Interface(n1=index, n2=1.0, R=-0.1),
        GRIN(d=0.002, g=2.0, n=1.0),
        repeated,
        input=source_for(kind),
    ).build()
    reference = packed_reference(simulation)
    theta = simulation.initial_values.copy()
    theta[simulation.parameter_info[id(d)].parameter_index] = 0.03
    before = theta.copy()
    result, expected = simulation.run(theta), reference.run(theta)

    numpy.testing.assert_array_equal(result.z, expected.z)
    assert result.z.dtype == expected.z.dtype
    assert result.element_info is simulation.element_info
    for state, original in zip(result.states, expected.states):
        assert type(state) is type(original)
        for field in fields(state):
            actual_value, expected_value = getattr(state, field.name), getattr(original, field.name)
            numpy.testing.assert_allclose(actual_value, expected_value, rtol=1e-12, atol=1e-14)
            assert numpy.asarray(actual_value).dtype == numpy.asarray(expected_value).dtype
    assert result.after(repeated, occurrence=1) is result.final
    assert result.z_after(repeated, occurrence=1) == expected.z_after(repeated, occurrence=1)

    fast = lambda t: observables(simulation.run(t))
    packed = lambda t: observables(reference.run(t))
    assert_finite_match(jacobian(fast)(theta), jacobian(packed)(theta), rtol=1e-9, atol=1e-10)
    cost = lambda t, evaluate: np.sum(evaluate(t) ** 2)
    assert_finite_match(grad(lambda t: cost(t, fast))(theta),
                        grad(lambda t: cost(t, packed))(theta), rtol=1e-9, atol=1e-10)
    assert_finite_match(hessian(lambda t: cost(t, fast))(theta),
                        hessian(lambda t: cost(t, packed))(theta), rtol=1e-9, atol=1e-10)
    numpy.testing.assert_array_equal(theta, before)
    assert simulation.parameter_info[id(d)].value == 0.025


@pytest.mark.parametrize("kind", ["gaussian", "ray", "bundle", "bundle32", "custom"])
def test_sequence_propagation_preserves_nested_and_fourth_order_autodiff(kind):
    distance = Parameter(0.025).variable()
    focal_length = Parameter(0.15).variable()
    simulation = System(
        Space(d=distance), ThinLens(f=focal_length), Space(d=distance ** 2 + 0.01),
        input=source_for(kind),
    ).build()
    reference = packed_reference(simulation)
    theta = simulation.initial_values.copy()

    def cost(t, run):
        values = observables(run(t))
        return np.sum(values ** 4)

    fast, packed = lambda t: cost(t, simulation.run), lambda t: cost(t, reference.run)
    assert_finite_match(jacobian(jacobian(hessian(fast)))(theta),
                        jacobian(jacobian(hessian(packed)))(theta), rtol=1e-9, atol=1e-9)
    nested = lambda t, function: np.sum(grad(function)(t) ** 2)
    assert_finite_match(hessian(lambda t: nested(t, fast))(theta),
                        hessian(lambda t: nested(t, packed))(theta), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", ["gaussian", "ray", "bundle", "bundle32", "custom"])
def test_solver_sequence_projection_composes_with_optical_states_and_composites(kind, monkeypatch):
    distance = Parameter(0.025).variable(bounds=(0.01, 0.06))
    focus = Parameter(0.15).variable(bounds=(0.08, 0.25))
    repeated = Space(d=distance)
    custom = CustomElement(shift=distance / 3, power=-0.001 / focus)
    simulation = System(
        repeated, custom, Slab(d=0.004, n=1.4),
        ThickLens(d=0.003, n=1.4, R1=0.1, R2=-0.1),
        Interface(n1=1.0, n2=1.4, R=0.1), Space(d=distance / 2, n=1.4),
        Interface(n1=1.4, n2=1.0, R=-0.1), ThinLens(f=focus), repeated,
        input=source_for(kind),
    ).build()
    field = "w" if kind == "gaussian" else "x"
    solver = Solver(simulation).target(
        Objective(lambda ctx: getattr(ctx.after(repeated, occurrence=1), field), weight=2.0),
        (distance / focus) ** 3,
        lambda ctx: ctx.z_after(custom) - 0.03,
    ).require(Constraint(lambda ctx: getattr(ctx.states[-1], field), lower_bound=-1., upper_bound=1.))

    def compile_functions():
        objectives = solver._compile_objectives()
        constraints = solver._compile_constraints(solver.constraints)
        def cost(theta):
            context = SolverContext(theta, simulation.run)
            return sum(obj.weight * np.sum(obj.evaluate(context) ** 2) for obj in objectives)
        def values(theta):
            context = SolverContext(theta, simulation.run)
            return np.concatenate([np.reshape(c.evaluate(context), (-1,)) for c in constraints])
        return cost, values

    with monkeypatch.context() as reference:
        reference.setattr(Solver, "_compile_node", packed_expression_reference)
        old_cost, old_values = compile_functions()
    cost, values = compile_functions()
    theta = simulation.initial_values.copy()
    for actual, expected in ((cost, old_cost), (values, old_values),
                             (grad(cost), grad(old_cost)), (jacobian(values), jacobian(old_values)),
                             (hessian(cost), hessian(old_cost)),
                             (jacobian(jacobian(hessian(cost))), jacobian(jacobian(hessian(old_cost))))):
        assert_finite_match(actual(theta), expected(theta), rtol=1e-9, atol=1e-9)
    weights = np.linspace(0.5, 1.5, len(values(theta)))
    assert_finite_match(hessian(lambda t: np.dot(weights, values(t)))(theta),
                        hessian(lambda t: np.dot(weights, old_values(t)))(theta), rtol=1e-9, atol=1e-9)
    nested = lambda t, f: np.sum(grad(f)(t) ** 2)
    assert_finite_match(hessian(lambda t: nested(t, cost))(theta),
                        hessian(lambda t: nested(t, old_cost))(theta), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", ["ray", "bundle"])
def test_scalar_float64_coefficients_keep_strong_promotion_for_float32_states(kind):
    source = (ParaxialRay(x=numpy.float32(0.1), theta=numpy.float32(0.2))
              if kind == "ray" else source_for("bundle32"))
    simulation = System(
        ABCD(A=1.0000000001, B=0.123456789123, C=-0.23456789123, D=0.98765432123),
        input=source,
    ).build()
    actual, expected = simulation.run(), packed_reference(simulation).run()

    for field in ("x", "theta"):
        result = getattr(actual.final, field)
        numpy.testing.assert_array_equal(result, getattr(expected.final, field))
        assert numpy.asarray(result).dtype == numpy.dtype("float64")


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part:numpy.exceptions.ComplexWarning")
def test_sequence_gaussian_aperture_keeps_complex_promotion_and_execution_bindings():
    radius = Parameter(1e-3).variable()
    simulation = System(
        Space(d=0.02), GaussianAperture(a=radius), Space(d=0.01),
        input=source_for("gaussian"),
    ).build()
    reference = packed_reference(simulation)
    theta = simulation.initial_values.copy()
    numpy.testing.assert_array_equal(simulation.run().q, reference.run().q)
    assert simulation.run().z.dtype == reference.run().z.dtype
    fast = lambda t: np.real(simulation.run(t).final.w)
    packed = lambda t: np.real(reference.run(t).final.w)
    assert_finite_match(grad(fast)(theta), grad(packed)(theta), rtol=1e-10)
    assert_finite_match(hessian(fast)(theta), hessian(packed)(theta), rtol=1e-10)


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part:numpy.exceptions.ComplexWarning")
def test_solver_complex_expression_preserves_beam_execution_bindings(monkeypatch):
    radius = Parameter(1e-3).variable()
    output = Space(d=0.01)
    simulation = System(Space(d=0.02), GaussianAperture(a=radius), output,
                        input=source_for("gaussian")).build()
    assert simulation.graph.symbols
    assert all(symbol.key in simulation.execution_context for symbol in simulation.graph.symbols)
    solver = Solver(simulation)
    expression = lambda ctx: radius ** 2 + ctx.after(output).q
    optimized = solver._compile_callable(expression)
    with monkeypatch.context() as reference:
        reference.setattr(Solver, "_compile_node", packed_expression_reference)
        original = solver._compile_callable(expression)

    def values(theta, evaluate):
        value = evaluate(SolverContext(theta, simulation.run))
        return np.stack((np.real(value), np.imag(value)))
    fast, old = lambda t: values(t, optimized), lambda t: values(t, original)
    theta = simulation.initial_values.copy()
    assert_finite_match(fast(theta), old(theta), rtol=1e-12)
    assert_finite_match(jacobian(fast)(theta), jacobian(old)(theta), rtol=1e-9)
    for differentiate in (grad, hessian, lambda f: jacobian(jacobian(hessian(f)))):
        assert_finite_match(differentiate(lambda t: np.sum(fast(t) ** 2))(theta),
                            differentiate(lambda t: np.sum(old(t) ** 2))(theta), rtol=1e-9, atol=1e-9)


def test_sequence_variable_refractive_index_transitions_match_packed_derivatives():
    index = Parameter(1.4).variable()
    simulation = System(
        Interface(n1=1.0, n2=index, R=0.1), Space(d=0.02, n=index),
        Interface(n1=index, n2=1.0, R=-0.1),
        input=source_for("gaussian"),
    ).build()
    reference = packed_reference(simulation)
    theta = simulation.initial_values.copy()
    fast = lambda t: observables(simulation.run(t))
    packed = lambda t: observables(reference.run(t))

    assert_finite_match(fast(theta), packed(theta), rtol=1e-12)
    assert_finite_match(jacobian(fast)(theta), jacobian(packed)(theta), rtol=1e-10)
    for derivative in (grad, hessian):
        assert_finite_match(derivative(lambda t: np.sum(fast(t) ** 2))(theta),
                            derivative(lambda t: np.sum(packed(t) ** 2))(theta), rtol=1e-9)


@pytest.mark.parametrize("binding", [numpy.float32(0.01), numpy.array(0.01), numpy.array([0.01, 0.02])])
def test_sequence_simulation_matches_packed_runtime_binding_coercion(binding):
    simulation = System(
        CustomElement(shift=0.0, power=Symbol("power")), input=source_for("custom"),
    ).build()
    simulation.execution_context = {"power": binding}
    reference = packed_reference(simulation)
    if numpy.ndim(binding):
        # Mixed vector/scalar graph roots were ragged before this optimization.
        for run in (reference.run, simulation.run):
            with pytest.raises(ValueError, match="inhomogeneous|sequence"):
                run()
    else:
        expected, actual = reference.run(), simulation.run()
        numpy.testing.assert_array_equal(observables(actual), observables(expected))
        assert numpy.asarray(actual.final.x).dtype == numpy.asarray(expected.final.x).dtype
