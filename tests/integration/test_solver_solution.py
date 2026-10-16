"""Solver orchestration, optimization, and complete solution reporting."""

from __future__ import annotations

import numpy as numpy
import pytest
from diffractix.graph import Parameter
from diffractix.beams import GaussianBeam
from diffractix.elements import Space
from diffractix.solver import Backend, Constraint, Objective, OptimizationResult, Solution, Solver
from diffractix.system import System
from tests.helpers.simulation import make_simulation


@pytest.mark.scipy
def test_solver_solve_returns_real_solution_with_scipy():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter - 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert isinstance(solution, Solution)
    assert solution.success
    assert solution.cost == pytest.approx(0.0, abs=1e-10)
    numpy.testing.assert_allclose(solution.x, [2.0], atol=1e-6)


@pytest.mark.scipy
def test_solver_solve_respects_parameter_bounds():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=3.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter - 10.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    numpy.testing.assert_allclose(solution.x, [3.0], atol=1e-6)


@pytest.mark.scipy
def test_solver_solve_enforces_hard_constraint():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(parameter)
    solver.require(parameter >= 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    assert solution.violations == ()
    numpy.testing.assert_allclose(solution.x, [2.0], atol=1e-6)
    assert solution.constraints[0].satisfied


@pytest.mark.scipy
def test_simulation_require_combines_with_system_and_solver_constraints():
    distance = Parameter(0.5).variable(bounds=(0.1, 1.0))
    space = Space(d=distance)
    beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    system = System(space, input=beam)
    system.require(Constraint(distance, lower_bound=0.2, label="system minimum"))
    simulation = system.build()
    solver = Solver(simulation)
    solver.target(distance - 0.9)
    solver.require(Constraint(distance, lower_bound=0.4, label="solver minimum"))
    original_system_requirements = system.requirements
    original_element_requirements = space.requirements
    original_graph = simulation.graph
    original_parameter_graph = simulation.parameter_graph
    built_requirements = simulation.requirements

    # Requirements added after Solver construction are picked up at solve time.
    simulation.require(
        distance >= 0.3,
        Constraint(
            lambda ctx: ctx.z_after(space),
            upper_bound=0.6,
            label="simulation maximum",
        ),
    )
    requirements = simulation.requirements

    for _ in range(2):
        solution = solver.solve(method="SLSQP")

        assert solution.success
        assert solution.feasible
        assert solution.violations == ()
        numpy.testing.assert_allclose(solution.x, [0.6], atol=1e-6)
        assert [item.constraint.label for item in solution.constraints] == [
            "system minimum", None, None, "simulation maximum", "solver minimum",
        ]
        numpy.testing.assert_allclose([item.value for item in solution.constraints], [0.6] * 5, atol=1e-6)

    assert len(built_requirements) == 2
    assert simulation.requirements is requirements
    assert system.requirements == original_system_requirements
    assert space.requirements == original_element_requirements
    assert simulation.graph is original_graph
    assert simulation.parameter_graph is original_parameter_graph
    assert distance.value == 0.5
    numpy.testing.assert_allclose(simulation.run().z, [0.0, 0.5])


@pytest.mark.scipy
def test_solver_solve_mixed_objectives_constraint_and_bounds():
    parameter = Parameter(
        1.0,
        name="p",
        variable=True,
        lower_bound=0.0,
        upper_bound=4.0,
    )
    solver = Solver(make_simulation((parameter,)))
    solver.target(
        Objective(parameter - 1.0),
        Objective(parameter - 3.0, weight=3.0),
    )
    solver.require(parameter >= 2.0)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    numpy.testing.assert_allclose(solution.x, [2.5], atol=1e-5)
    assert solution.cost == pytest.approx(sum(target.cost for target in solution.targets))


def test_reduced_optimizer_constraints_do_not_erase_physical_solution_reporting(monkeypatch):
    from diffractix.solver import solver as solver_module

    d1 = Parameter(0.006).variable(bounds=(0.004, 0.010))
    d2 = Parameter(0.010).variable(bounds=(0.005, 0.014))
    d3 = 0.025 - d1 - d2
    beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6)
    simulation = System(Space(d=d1), Space(d=d2), Space(d=d3), input=beam).build()
    solver = Solver(simulation)
    solver.target(d1 - 0.007, d2 - 0.010)
    solver.require(
        Constraint(d3, lower_bound=0.004, label="clearance"),
        Constraint(d1, lower_bound=0.003, label="bound-redundant"),
    )
    requirements = simulation.requirements
    explicit = tuple(solver.constraints)
    captured = []

    def backend(problem, method=None, options=None):
        captured.append(problem.unscale())
        return OptimizationResult(
            x=problem.x0,
            success=True,
            cost=problem.objective(problem.x0),
            message="initial feasible point",
            scaling=problem.scaling,
        )

    monkeypatch.setattr(solver_module, "solve_scipy", backend)
    for _ in range(2):
        solution = solver.solve()
        physical = captured[-1]
        assert physical.n_constraints == 1
        numpy.testing.assert_allclose(physical.constraints(solution.x), [0.009])
        numpy.testing.assert_allclose(physical.constraint_lower, [0.004])
        assert len(solution.constraints) == 5
        assert [result.constraint.label for result in solution.constraints] == [None, None, None, "clearance", "bound-redundant"]
        numpy.testing.assert_allclose([result.value for result in solution.constraints], [0.006, 0.010, 0.009, 0.009, 0.006])
        assert solution.feasible
        assert solution.violations == ()

    assert simulation.requirements is requirements
    assert all(current is original for current, original in zip(solver.constraints, explicit))
    assert len(solver.constraints) == len(explicit)
    assert d1.value == 0.006
    assert d2.value == 0.010
