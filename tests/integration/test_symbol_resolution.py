"""Compiled and interpreted consumers should agree on Simulation bindings."""
import numpy as np
import pytest

from diffractix import Constraint, GaussianBeam, Parameter, Ray, Solver, Symbol, System
from diffractix.backend import xp
from diffractix.backend import grad, hessian
from diffractix.graph import Literal
from diffractix.solver.context import SolverContext
from diffractix.system import WAVELENGTH


def bound_source(kind):
    parameter = Parameter(0.02).variable(bounds=(-0.1, 0.1))
    binding = {
        "scalar": 0.02,
        "literal": Literal(0.02),
        "fixed": Parameter(0.02),
        "variable": parameter,
        "expression": 2 * parameter + 0.01,
        "alias": Symbol("base"),
    }[kind]
    source = Ray(x=Parameter(0.1).variable(bounds=(-1.0, 1.0)), theta=Symbol("angle"))
    system = System(input=source).add_context("base", parameter).add_context("angle", binding)
    return source, system.build()


@pytest.mark.scipy
@pytest.mark.parametrize("kind", ["scalar", "literal", "fixed", "variable", "expression", "alias"])
def test_solution_value_of_resolves_the_same_compile_bindings_as_static_objectives(kind):
    source, simulation = bound_source(kind)
    solution = Solver(simulation).target(source.x - 0.1).solve(method="SLSQP", use_jit=False)
    assert solution.success and solution.feasible
    expected = solution.run().source.theta
    static = Solver(simulation)._compile_node(Symbol("angle"))
    assert static(SolverContext(solution.x, simulation.run)) == pytest.approx(expected)

    assert solution.value_of(Symbol("angle")) == pytest.approx(expected)


@pytest.mark.scipy
def test_solution_value_of_resolves_the_compiled_source_wavelength():
    source = GaussianBeam(q=1j, wavelength=Parameter(1e-6).variable(bounds=(0.5e-6, 2e-6)), n=1.0)
    simulation = System(input=source).build()
    solution = Solver(simulation).target(source.wavelength / 1e-6 - 1).solve(method="SLSQP", use_jit=False)
    assert solution.success and solution.feasible
    assert solution.run().source.wavelength == pytest.approx(1e-6)

    assert solution.value_of(WAVELENGTH) == pytest.approx(1e-6)


@pytest.mark.parametrize("consumer", ["objective", "constraint"])
@pytest.mark.parametrize("kind", ["variable", "expression"])
def test_dynamic_ast_branches_resolve_compile_bindings(consumer, kind):
    source, simulation = bound_source(kind)

    def dynamic(context):
        if context.states[0].x > 0:
            return Symbol("angle")**2
        return -Symbol("angle")**2

    solver = Solver(simulation)
    if consumer == "objective":
        solver.target(dynamic)
        compiled, = solver._compile_objectives(solver.objectives)
    else:
        compiled, = solver._compile_constraints((Constraint(dynamic, lower_bound=-1.0),))

    source_index = simulation.parameter_info[id(source.x.node)].parameter_index
    for position in (0.1, -0.1):
        point = simulation.initial_values.copy()
        point[source_index] = position
        point = xp.array(point)
        context = SolverContext(point, simulation.run)
        sign = 1 if position > 0 else -1
        expected = context.states[0].theta**2 * sign
        assert compiled.evaluate(context) == pytest.approx(expected)
        evaluate = lambda theta: compiled.evaluate(SolverContext(theta, simulation.run))
        # Away from the Python branch boundary, the bound Parameter remains
        # differentiable through its canonical theta index in either branch.
        expected_gradient = np.zeros(len(point))
        bound_index = simulation.parameter_info[id(simulation.compile_context["base"])].parameter_index
        factor = 2.0 if kind == "expression" else 1.0
        expected_gradient[bound_index] = 2 * context.states[0].theta * factor * sign
        np.testing.assert_allclose(grad(evaluate)(point), expected_gradient)
        expected_hessian = np.zeros((len(point), len(point)))
        expected_hessian[bound_index, bound_index] = 2 * factor**2 * sign
        np.testing.assert_allclose(hessian(evaluate)(point), expected_hessian, atol=1e-14)
