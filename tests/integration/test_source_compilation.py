"""One graph and canonical theta for declarative sources and optical elements."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from diffractix import (
    GaussianAperture, GaussianBeam, Interface, Node, Parameter, ParaxialState,
    Ray, RayBundle, Solver, Space, Symbol, System, ThinLens,
)
from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.graph import CompiledAST, Literal, compile_ast
from diffractix.graph.utils import ASTCycleError
from diffractix.simulation import Simulation
from diffractix.simulation.simulation import SimulationSource
from diffractix.solver.context import SolverContext
from diffractix.system import AMBIENT_N, WAVELENGTH


def test_source_only_variables_reconstruct_each_run_without_compiling_again(monkeypatch):
    source = Ray(x=1e-3, theta=0.02).variable("x", "theta")
    simulation = System(input=source).build()
    assert simulation.graph.variables == (source.x.node, source.theta.node)
    assert simulation._source_spec.field_indices == (("x", 0), ("theta", 1), ("n", 2))
    calls = []
    evaluate = CompiledAST._evaluate_roots

    def counted(self, *args, **kwargs):
        calls.append(self)
        return evaluate(self, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Running must not compile or invoke numeric()")

    monkeypatch.setattr(CompiledAST, "_evaluate_roots", counted)
    monkeypatch.setattr("diffractix.input.input.compile_ast", forbidden)
    monkeypatch.setattr("diffractix.system.system.compile_ast", forbidden)
    monkeypatch.setattr(Ray, "numeric", forbidden)
    first = simulation.run(xp.array([2e-3, -0.01]))
    second = simulation.run()
    assert calls == [simulation.graph, simulation.graph]
    assert first.source is first.initial is first.final
    assert first.source is not second.source is not simulation.source
    np.testing.assert_allclose(first.source.vector, [2e-3, -0.01])
    np.testing.assert_allclose(second.source.vector, [1e-3, 0.02])
    np.testing.assert_allclose(simulation.source.vector, [1e-3, 0.02])
    assert type(first.source) is source.numerical_type()
    assert simulation.parameter_info[id(source.x.node)].owner_type == "Ray"


def test_source_and_element_parameters_share_theta_but_use_output_indices():
    distance = Parameter(0.025).variable()
    source = Ray(x=1e-3, theta=0.02).variable("x", "theta")
    space = Space(d=distance)
    simulation = System(space, input=source).build()
    assert simulation.graph.variables == (distance, source.x.node, source.theta.node)
    assert simulation.steps[0].matrix_indices == ((0, 1), (2, 3))
    assert simulation.steps[0].length_index == 4
    assert simulation._source_spec.field_indices == (("x", 6), ("theta", 7), ("n", 8))
    for index, parameter in enumerate(simulation.graph.variables):
        assert simulation.parameter_info[id(parameter)].parameter_index == index
    point = xp.array([0.04, 2e-3, -0.01])
    result = simulation.run(point)
    np.testing.assert_allclose(result.final.vector, [1.6e-3, -0.01])
    assert all(type(state) is source.numerical_type() for state in result.states)
    values = simulation.parameter_graph.evaluate(simulation.parameter_graph.initial_values)
    np.testing.assert_allclose(values, [0.025, 1e-3, 0.02, 1.0])


def test_source_and_optics_can_share_one_independent_parameter():
    shared = Parameter(0.02).variable()
    source = Ray(x=3 * shared + 0.001, theta=2 * shared)
    simulation = System(Space(d=shared), input=source).build()
    assert simulation.graph.variables == (shared,)
    result = simulation.run(xp.array([0.03]))
    np.testing.assert_allclose(result.source.vector, [0.091, 0.06])
    np.testing.assert_allclose(result.final.vector, [0.0928, 0.06])


@pytest.mark.parametrize("binding", [None, "shared"])
def test_source_inherits_or_reuses_ambient_parameter_without_extra_equality(binding):
    source = Ray(x=1e-3, theta=0.02)
    system = System(input=source, ambient_n=1.2, ambient_n_variable=True)
    if binding:
        source.n = system.ambient_n
    simulation = system.build()
    assert simulation.graph.variables == (system.ambient_n,)
    assert simulation.requirements == ()
    assert simulation.source.refractive_index == pytest.approx(1.2)
    assert simulation.run(xp.array([1.4])).source.refractive_index == pytest.approx(1.4)
    assert simulation.compile_context[AMBIENT_N.key] is system.ambient_n


def test_mismatched_source_medium_fails_during_build():
    with pytest.raises(ValueError, match="Refractive index mismatch"):
        System(input=Ray(x=0.0, theta=0.0, n=1.5)).build()


@pytest.mark.scipy
def test_independent_source_and_ambient_indices_remain_equal_during_optimization():
    source = Ray(x=0.0, theta=0.0, n=Parameter(1.2).variable(bounds=(1.0, 2.0)))
    system = System(input=source, ambient_n=1.2, ambient_n_variable=True)
    system.ambient_n.bound(1.0, 2.0)
    simulation = system.build()
    continuity, = simulation.requirements
    assert continuity.lower_bound == continuity.upper_bound == 0.0
    assert "Ray" in continuity.label
    solution = Solver(simulation).target(source.n - 1.8, system.ambient_n - 1.4).solve(
        method="SLSQP", options={"ftol": 1e-12},
    )
    assert solution.success, solution.message
    assert solution.feasible
    assert solution[source.n.node] == pytest.approx(1.6, abs=1e-8)
    assert solution[system.ambient_n] == pytest.approx(1.6, abs=1e-8)


@pytest.mark.scipy
def test_source_requirements_reach_the_solver_without_mutating_the_input():
    source = Ray(x=0.1, theta=0.0).variable("x")
    source.x.node.bound(-1.0, 1.0)
    requirement = source.x >= 0.25
    source.require(requirement)
    system = System(input=source)
    simulation = system.build()
    solution = Solver(simulation).target(source.x).solve(method="SLSQP")
    assert solution.success, solution.message
    assert solution[source.x.node] == pytest.approx(0.25, abs=1e-8)
    assert len(solution.constraints) == 1
    assert simulation.requirements[0].lower_bound == 0.25
    assert source.requirements == (requirement,)
    assert system.requirements == ()
    assert source.x.value == 0.1


@pytest.mark.parametrize("dimension", ["source", "combined"])
@pytest.mark.filterwarnings("ignore:Output seems independent of input\\.:UserWarning")
def test_source_propagation_analytical_derivatives_and_jit(dimension):
    source = Ray(x=1e-3, theta=0.02).variable("x", "theta")
    distance = Parameter(0.04, variable=dimension == "combined")
    simulation = System(Space(d=distance), ThinLens(f=0.2), input=source).build()

    def trace(theta):
        return simulation.run(theta).final.vector

    if dimension == "combined":
        expected_jacobian = [[0.02, 1, 0.04], [-0.1, -5, 0.8]]
        expected_hessian = np.zeros((3, 3))
        expected_hessian[0, 2] = expected_hessian[2, 0] = 1.0
    else:
        expected_jacobian = [[1, 0.04], [-5, 0.8]]
        expected_hessian = np.zeros((2, 2))
    initial = simulation.initial_values
    for fn, expected in (
        (trace, [0.0018, 0.011]),
        (jacobian(trace), expected_jacobian),
        (grad(lambda t: trace(t)[0]), expected_jacobian[0]),
        (hessian(lambda t: trace(t)[0]), expected_hessian),
    ):
        np.testing.assert_allclose(fn(initial), expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(jit(fn)(xp.array(initial)), expected, rtol=1e-12, atol=1e-14)
    energy = lambda t: xp.sum(trace(t)**2)
    nested = lambda t: xp.sum(grad(energy)(t)**2)
    np.testing.assert_allclose(hessian(nested)(initial), jit(hessian(nested))(xp.array(initial)),
                               rtol=1e-11, atol=1e-12)


def test_variable_wavelength_stays_real_alongside_complex_optics_and_is_differentiable():
    wavelength = Parameter(1.064e-6).variable(bounds=(0.5e-6, 2e-6))
    source = GaussianBeam(q=0.01 + 0.03j, wavelength=wavelength)
    aperture = GaussianAperture(a=1e-3)
    system = System(aperture, input=source)
    simulation = system.build()
    assert simulation.graph.variables == (wavelength,)
    assert simulation.graph.symbols == ()
    assert WAVELENGTH.key in simulation.compile_context
    assert WAVELENGTH.key not in simulation.execution_context
    assert WAVELENGTH.key not in system.compile_context
    assert np.isrealobj(simulation.source.wavelength)

    def values(t):
        result = simulation.run(t)
        return xp.stack((result.source.wavelength, result.final.wavelength, xp.real(result.final.q),
                         xp.imag(result.final.q), result.final.w))

    def analytical(t):
        lam = t[0]
        inverse_q = 1 / source.q - 1j * lam / (np.pi * 1e-6)
        q = 1 / inverse_q
        width = xp.sqrt(-lam / (np.pi * xp.imag(inverse_q)))
        return xp.stack((lam, lam, xp.real(q), xp.imag(q), width))

    for point in (xp.array([0.7e-6]), xp.array([1.5e-6])):
        assert np.isrealobj(simulation.run(point).source.wavelength)
        assert np.iscomplexobj(simulation.run(point).final.q)
        for transform in (lambda f: f, jacobian, lambda f: hessian(lambda t: xp.sum(f(t)**2))):
            fn, reference = transform(values), transform(analytical)
            np.testing.assert_allclose(fn(point), reference(point), rtol=2e-11, atol=1e-10)
            np.testing.assert_allclose(jit(fn)(point), reference(point), rtol=2e-11, atol=1e-10)
    solver = Solver(simulation).target(Symbol(WAVELENGTH.key) / 1e-6)
    objective, = solver._compile_objectives(solver.objectives)
    assert objective.evaluate(SolverContext(xp.array([1.5e-6]), simulation.run)) == pytest.approx(1.5)


def test_source_binding_conflicts_are_explicit_and_shared_binding_is_allowed():
    source = GaussianBeam(q=0.02j, wavelength=1e-6)
    system = System(input=source)
    system.add_context(WAVELENGTH.key, source.wavelength.node)
    assert system.build().source.wavelength == pytest.approx(1e-6)
    system.add_context(WAVELENGTH.key, 2e-6)
    with pytest.raises(ValueError, match="Conflicting.*wavelength"):
        system.build()


def test_fixed_and_derived_source_fields_are_snapshotted_without_live_handles():
    base = Parameter(1e-3)
    source = Ray(x=2 * base, theta=0.02)
    simulation = System(Space(d=0.04), input=source).build()
    before = simulation.run()
    base.value = 0.5
    source.theta = 10.0
    source.x = 9.0
    after = simulation.run()
    np.testing.assert_allclose(before.source.vector, [0.002, 0.02])
    np.testing.assert_array_equal(after.vector, before.vector)
    assert simulation.parameter_info[id(base)].value == 0.001
    assert not simulation.source.parameter_names


def test_source_symbols_use_system_bindings_and_parameter_snapshots():
    position = Parameter(0.002).variable()
    source = Ray(x=Symbol("launch_x"), theta=0.01)
    system = System(Space(d=0.1), input=source).add_context("launch_x", position)
    simulation = system.build()
    position.fixed()
    position.value = 9.0
    source.x = 42.0
    np.testing.assert_allclose(simulation.run().source.vector, [0.002, 0.01])
    np.testing.assert_allclose(simulation.run(xp.array([0.003])).final.vector, [0.004, 0.01])
    assert simulation.parameter_info[id(position)].parameter_index == 0
    with pytest.raises(KeyError, match="launch_x"):
        System(input=Ray(x=Symbol("launch_x"), theta=0.0)).build()


def test_input_without_graph_fields_supports_an_empty_compilation():
    @dataclass
    class PlainInput(ParaxialState):
        value: float
        doubled: float = field(init=False)

        def __post_init__(self):
            self.doubled = 2 * self.value

        def propagate(self, A, B, C, D, n):
            return type(self)(value=A * self.value + B)

    source = PlainInput(2.0)
    source.doubled = -999.0
    simulation = System(input=source).build()
    result = simulation.run()
    assert simulation.graph.variables == simulation.graph.program.root_indices == ()
    assert simulation._source_spec.field_indices == ()
    assert simulation.source.doubled == result.source.doubled == 4.0
    assert result.source is not simulation.source
    assert type(result.source) is source.numerical_type()


@dataclass(eq=False)
class CustomInput(ParaxialState):
    position: Node
    medium: Node
    optional: Node | None = None
    samples: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]))
    doubled: Node = field(init=False)

    def __post_init__(self):
        self.doubled = 2 * self.position

    @property
    def refractive_index(self):
        return self.medium

    def propagate(self, A, B, C, D, n):
        return type(self)(position=A * self.position + B, medium=n, optional=self.optional,
                          samples=self.samples)


def test_custom_input_constructor_fields_recompute_caches_and_snapshot_arrays():
    source = CustomInput(position=Parameter(0.1).variable(), medium=Symbol(AMBIENT_N.key))
    source.doubled = Symbol("stale_cache")
    samples = source.samples.copy()
    simulation = System(Space(d=0.2), input=source, ambient_n=1.5).build()
    source.samples[:] = 99.0
    source.position = 42.0
    result = simulation.run(xp.array([0.3]))
    assert result.source.position == pytest.approx(0.3)
    assert result.source.doubled == pytest.approx(0.6)
    assert result.final.position == pytest.approx(0.5)
    assert result.final.doubled == pytest.approx(1.0)
    assert result.source.medium == pytest.approx(1.5)
    assert result.source.optional is None
    assert all(type(s) is source.numerical_type() for s in result.states)
    np.testing.assert_array_equal(result.source.samples, samples)
    np.testing.assert_array_equal(simulation.source.samples, samples)
    assert not np.shares_memory(source.samples, result.source.samples)
    assert len(simulation.requirements) == 1  # Space.d >= 0; no source continuity equality.
    assert simulation.requirements[0].upper_bound == np.inf


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numerical_bundle_snapshot_retains_array_dtype_with_real_coefficients(dtype):
    source = RayBundle.numerical_type()(x=xp.array([0.001, -0.002], dtype=dtype),
                                       theta=xp.array([0.01, -0.02], dtype=dtype), n=1.0)
    simulation = System(Space(d=0.025), input=source).build()
    result = simulation.run()
    assert result.source.x.dtype == np.dtype(dtype)
    assert result.final.x.dtype == np.dtype("float64")
    expected = np.asarray(source.x, dtype="float64") + 0.025 * np.asarray(source.theta, dtype="float64")
    np.testing.assert_allclose(result.final.x, expected, rtol=1e-14)
    np.testing.assert_allclose(jit(lambda t: simulation.run(t).final.x)(xp.array([])), expected, rtol=1e-14)


@pytest.mark.parametrize("target", ["original", "initial_snapshot", "previous_result"])
def test_mutable_source_arrays_cannot_contaminate_other_runs(target):
    source = RayBundle(x=np.array([0.001, -0.002]), theta=np.array([0.01, -0.02]))
    simulation = System(Space(d=0.1), ThinLens(f=0.2), input=source).build()
    first, second = simulation.run(), simulation.run()
    expected = np.array(second.vectors, copy=True)
    objects = {"original": source, "initial_snapshot": simulation.source, "previous_result": first.source}
    objects[target].x[0] = 100.0
    objects[target].theta[:] = -100.0
    np.testing.assert_array_equal(second.vectors, expected)
    np.testing.assert_array_equal(simulation.run().vectors, expected)
    assert not np.shares_memory(first.source.x, second.source.x)
    assert not np.shares_memory(simulation.source.x, second.source.x)


def test_propagated_mutable_fields_do_not_expose_source_spec_storage():
    source = CustomInput(position=0.1, medium=Symbol(AMBIENT_N.key))
    simulation = System(Space(d=0.2), input=source).build()
    result = simulation.run()
    result.final.samples[:] = 100.0
    np.testing.assert_array_equal(simulation.source.samples, [1.0, 2.0])
    np.testing.assert_array_equal(simulation.run().final.samples, [1.0, 2.0])


def test_source_and_element_handle_mutations_preserve_the_compiled_program():
    source = Ray(x=0.001, theta=0.02).variable("x")
    space, lens = Space(d=0.1).variable("d"), ThinLens(f=0.2)
    simulation = System(space, lens, input=source).build()
    before = np.array(simulation.run().vector, copy=True)
    initial = simulation.initial_values.copy()
    source.x.node.fixed()
    source.x.node.value = 0.5
    source.x = 42.0
    source.theta = -99.0
    space.d.node.fixed()
    space.d.node.value = 20.0
    space.d = 50.0
    lens.f = 0.01
    np.testing.assert_array_equal(simulation.run().vector, before)
    np.testing.assert_array_equal(simulation.initial_values, initial)


def test_source_spec_owns_fixed_fields_and_snapshot_uses_run_normalization():
    positions = np.array([0.001, -0.002])
    fixed = SimulationSource(RayBundle.numerical_type(), (("x", positions), ("theta", positions), ("n", 1.0)), ())
    positions[:] = 99.0
    first = fixed.construct(())
    first.x[:] = -99.0
    np.testing.assert_array_equal(fixed.construct(()).x, [0.001, -0.002])

    graph = compile_ast((Literal(1j), Parameter(0.002).variable(), Literal(0.01), Literal(1.0)))
    spec = SimulationSource(Ray.numerical_type(), (), (("x", 1), ("theta", 2), ("n", 3)))
    simulation = Simulation(
        source_spec=spec, graph=graph, steps=(), parameter_info={}, compile_context={},
        execution_context={}, location_map={}, requirements=(),
        parameter_graph=compile_ast(()), element_info=(),
    )
    assert isinstance(simulation._source_spec, SimulationSource)
    result = simulation.run(xp.array([0.003]))
    assert simulation.source.x == pytest.approx(0.002)
    assert result.source.x == pytest.approx(0.003)
    for name in ("x", "theta", "n"):
        initial_value, current_value = getattr(simulation.source, name), getattr(result.source, name)
        assert np.isrealobj(initial_value) and np.isrealobj(current_value)
        assert np.asarray(initial_value).dtype == np.asarray(current_value).dtype
        assert np.shape(initial_value) == np.shape(current_value) == ()


def test_entrance_continuity_uses_ambient_not_first_interface_output():
    source = Ray(x=0.001, theta=0.02)
    interface = Interface(n1=1.2, n2=1.5)
    simulation = System(interface, input=source, ambient_n=1.2).build()
    result = simulation.run()
    assert result.source.refractive_index == pytest.approx(1.2)
    assert result.final.refractive_index == pytest.approx(1.5)
    assert result.final.theta == pytest.approx(0.02 * 1.2 / 1.5)
    assert len(simulation.requirements) == 3  # Two index domains and interface continuity.


def test_model_level_index_aliases_are_resolved_without_compiling_a_graph(monkeypatch):
    system = System(ambient_n_variable=True)
    source = CustomInput(position=0.1, medium=Symbol("entrance_index"))
    system.add_input_beam(source).add_context("entrance_index", Symbol(AMBIENT_N.key))

    def forbidden(*args, **kwargs):
        raise AssertionError("Source identity checks must not compile or inspect optimized programs")

    with monkeypatch.context() as patch:
        patch.setattr("diffractix.system.system.compile_ast", forbidden)
        assert system._source_continuity(system.compile_context) == ()
    simulation = system.build()
    assert simulation.requirements == ()
    assert simulation.run(xp.array([1.4])).source.medium == pytest.approx(1.4)


def test_cyclic_index_bindings_fail_clearly_instead_of_looping():
    system = System(input=Ray(x=0.0, theta=0.0, n=Symbol("cycle")))
    system.add_context("cycle", Symbol("cycle"))
    with pytest.raises(ASTCycleError, match="Cycle"):
        system.build()


def test_fixed_numerical_wavelength_is_a_frozen_compiled_binding():
    wavelength = np.array(1e-6)
    source = GaussianBeam.numerical_type()(q=0.02j, wavelength=wavelength, n=1.0)
    simulation = System(GaussianAperture(a=0.001), input=source).build()
    before = np.array(simulation.run().q, copy=True)
    wavelength[...] = 2e-6
    source.wavelength = 3e-6
    assert simulation.graph.symbols == ()
    assert simulation.execution_context == {}
    assert simulation.compile_context[WAVELENGTH.key].value == 1e-6
    np.testing.assert_array_equal(simulation.run().q, before)


def test_runtime_array_source_bindings_remain_supported_and_owned():
    @dataclass
    class ArrayContextInput(ParaxialState):
        x: np.ndarray
        coefficient: np.ndarray

        @property
        def execution_context(self):
            return {"coefficient": self.coefficient}

        def propagate(self, A, B, C, D, n):
            return type(self)(x=A * self.x + B, coefficient=self.coefficient)

    from diffractix import ABCD

    coefficient = np.array([0.2, 0.3])
    source = ArrayContextInput(np.array([1.0, 2.0]), coefficient)
    simulation = System(ABCD(A=Symbol("coefficient")), input=source).build()
    coefficient[:] = 99.0
    np.testing.assert_allclose(simulation.run().final.x, [0.2, 0.6])
    assert tuple(symbol.key for symbol in simulation.graph.symbols) == ("coefficient",)
    assert "coefficient" not in simulation.compile_context
    np.testing.assert_array_equal(simulation.execution_context["coefficient"], [0.2, 0.3])
    # Explicit runtime replacement stays numerical; no recompilation is needed.
    simulation.execution_context = {"coefficient": xp.array([0.4, 0.5])}
    np.testing.assert_allclose(jit(lambda t: simulation.run(t).final.x)(xp.array([])), [0.4, 1.0])
