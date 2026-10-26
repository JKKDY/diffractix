"""Build-time snapshots must also survive later solver compilation/reduction."""
import numpy as np
import pytest

from diffractix import Constraint, Parameter, Ray, Solver, Space, Symbol, System
from diffractix.graph import Literal
from diffractix.backend import xp
from diffractix.solver.context import SolverContext


@pytest.mark.parametrize("owner", ["source", "element", "system"])
@pytest.mark.parametrize("mutation", ["parameter_state", "handle_target"])
def test_built_requirement_keeps_its_original_parameter_reference(owner, mutation):
    parameter = Parameter(0.1).variable(bounds=(0.0, 1.0))
    source = Ray(x=parameter if owner == "source" else 0.0, theta=0.0)
    space = Space(d=parameter if owner != "source" else 0.01)
    system = System(space, input=source)
    target = source if owner == "source" else space
    field = "x" if owner == "source" else "d"
    requirement = Constraint(2 * getattr(target, field) + 0.01, lower_bound=0.05, label="snapshot")
    (system if owner == "system" else target).require(requirement)
    simulation = system.build()
    stored = next(c for c in simulation.requirements if c.label == "snapshot")
    point = xp.array([0.3])
    context = SolverContext(point, simulation.run)
    before = Solver(simulation)._compile_constraints((stored,))[0]
    assert before.evaluate(context) == pytest.approx(0.61)

    if mutation == "parameter_state":
        parameter.value = 0.8
        parameter.bound(0.7, 0.9)
        parameter.fixed()
    else:
        setattr(target, field, 0.8)

    # Already-compiled evaluation/propagation remain stable in both cases.
    assert before.evaluate(context) == pytest.approx(0.61)
    result = simulation.run(point)
    propagated = result.source.x if owner == "source" else result.z[-1]
    assert propagated == pytest.approx(0.3)
    after = Solver(simulation)._compile_constraints((stored,))[0]
    assert after.evaluate(context) == pytest.approx(0.61)


@pytest.mark.parametrize("compound", [False, True], ids=["handle", "expression"])
def test_context_ast_keeps_build_time_handles_for_fresh_solver_compilation(compound):
    source = Ray(x=Parameter(0.1).variable(), theta=0.0)
    binding = 2 * source.x + 0.01 if compound else source.x
    simulation = System(input=source).add_context("position", binding).build()
    point = xp.array([0.3])
    context = SolverContext(point, simulation.run)
    expected = 0.61 if compound else 0.3
    before = Solver(simulation)._compile_node(Symbol("position"))
    assert before(context) == pytest.approx(expected)

    source.x = 0.8

    assert before(context) == pytest.approx(expected)
    assert simulation.run(point).source.x == pytest.approx(0.3)
    after = Solver(simulation)._compile_node(Symbol("position"))
    assert after(context) == pytest.approx(expected)


def test_requirements_and_context_share_an_independent_snapshot_of_the_same_ast():
    parameter = Parameter(0.1).variable()
    source = Ray(x=parameter, theta=0.0)
    expression = 2 * source.x + 0.01
    system = System(input=source).add_context("position", expression)
    system.require(
        Constraint(expression, lower_bound=0.0, label="lower"),
        Constraint(expression, upper_bound=1.0, label="upper"),
    )
    simulation = system.build()
    first, second = simulation.requirements
    assert first.evaluate is second.evaluate is simulation.compile_context["position"]
    assert first.evaluate is not expression
    assert simulation.graph.variables == (parameter,)
    assert [c.label for c in simulation.requirements] == ["lower", "upper"]

    expression.right = Literal(99.0)
    source.x = 0.8
    context = SolverContext(xp.array([0.3]), simulation.run)
    compiled = Solver(simulation)._compile_constraints(simulation.requirements)
    assert [c.evaluate(context) for c in compiled] == pytest.approx([0.61, 0.61])
    assert Solver(simulation)._compile_node(Symbol("position"))(context) == pytest.approx(0.61)


@pytest.mark.parametrize("variable", [False, True], ids=["fixed", "external-variable"])
def test_unused_context_parameter_state_is_also_captured_without_adding_theta(variable):
    external = Parameter(0.02, name="external", variable=variable)
    source = Ray(x=Parameter(0.1).variable(), theta=0.0)
    simulation = System(input=source).add_context("offset", external).build()
    assert simulation.graph.variables == (source.x.node,)
    external.value = 99.0
    external.fixed() if variable else external.variable()

    solver = Solver(simulation)
    if variable:
        with pytest.raises(ValueError, match="not part of this Simulation"):
            solver._compile_node(Symbol("offset"))
    else:
        context = SolverContext(simulation.initial_values, simulation.run)
        assert solver._compile_node(Symbol("offset"))(context) == pytest.approx(0.02)


def test_requirement_added_to_simulation_captures_its_handle_at_attachment():
    source = Ray(x=Parameter(0.1).variable(), theta=0.0)
    simulation = System(input=source).build()
    simulation.require(source.x >= 0.05)
    source.x = 0.8

    constraint, = Solver(simulation)._compile_constraints(simulation.requirements)
    assert constraint.evaluate(SolverContext(xp.array([0.3]), simulation.run)) == pytest.approx(0.3)


def test_generated_continuity_keeps_independent_source_and_ambient_indices():
    source_index = Parameter(1.2).variable(bounds=(1.0, 2.0))
    source = Ray(x=0.0, theta=0.0, n=source_index)
    system = System(input=source, ambient_n=1.2, ambient_n_variable=True)
    system.ambient_n.bound(1.0, 2.0)
    simulation = system.build()
    continuity, = simulation.requirements
    assert continuity.lower_bound == continuity.upper_bound == 0.0
    point = simulation.initial_values.copy()
    point[simulation.parameter_info[id(source_index)].parameter_index] = 1.5
    point[simulation.parameter_info[id(system.ambient_n)].parameter_index] = 1.3
    context = SolverContext(xp.array(point), simulation.run)
    before, = Solver(simulation)._compile_constraints((continuity,))
    assert before.evaluate(context) == pytest.approx(0.2)

    source.n = system.ambient_n

    assert before.evaluate(context) == pytest.approx(0.2)
    assert simulation.run(point).source.n == pytest.approx(1.5)
    after, = Solver(simulation)._compile_constraints((continuity,))
    assert after.evaluate(context) == pytest.approx(0.2)


@pytest.mark.scipy
@pytest.mark.parametrize("replacement", ["new_fixed", "snapshotted_fixed"])
def test_live_handle_change_cannot_remove_a_cached_solvers_required_constraint(replacement):
    parameter = Parameter(0.003).variable(bounds=(-0.01, 0.01))
    source = Ray(x=parameter, theta=0.008)
    source.require(source.x >= 0.0025)
    simulation = System(input=source).build()
    solver = Solver(simulation).target(parameter / 0.001)
    first = solver.solve(method="SLSQP", use_jit=False, options={"ftol": 1e-12})
    assert first.success and first.feasible
    assert first[parameter] == pytest.approx(0.0025, abs=1e-8)
    cache = solver._cache

    source.x = 0.008 if replacement == "new_fixed" else source.theta
    second = solver.solve(method="SLSQP", use_jit=False, options={"ftol": 1e-12})

    assert solver._cache is cache
    assert second.success, second.message
    np.testing.assert_allclose(second.x, first.x, atol=1e-8)
    assert second.feasible
