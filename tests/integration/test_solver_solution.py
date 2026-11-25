"""Real optimizer termination, full physical reporting, and solve overrides."""
import numpy as np
import pytest
from diffractix import Constraint, Parameter, Ray, Solver, Space, System
from diffractix.graph import Symbol


@pytest.mark.parametrize("exact_match", [True, False])
def test_reporting_only_reuses_constraints_at_the_exact_returned_point(monkeypatch, exact_match):
    from diffractix.solver import solver as solver_module
    from diffractix.solver.solution import SolverResult

    p = Parameter(0.5).variable(bounds=(0, 1))
    simulation = System(Space(d=1), source=Ray(x=p, theta=0.1, n=1)).build()
    solver = Solver(simulation).target(p - 0.5)
    solver.require(p >= 0, simulation.symbolic.final.x <= 0.7)

    def solve(problem, **kwargs):
        problem.constraints(problem.x0)
        point = problem.x0 if exact_match else problem.x0 + 0.5
        return SolverResult(point, True, 0, "stub")

    monkeypatch.setattr(solver_module, "solve_scipy", solve)
    solution = solver.solve(optimizer="scipy")
    assert "constraints" not in solution.__dict__
    original_run = solution.run
    calls = []

    def record_run():
        calls.append(1)
        return original_run()

    monkeypatch.setattr(solution, "run", record_run)
    assert solution.feasible is exact_match
    assert len(solution.constraints) == 2
    assert calls == ([] if exact_match else [1])


@pytest.mark.parametrize("use_jit", [False, True])
def test_bound_relaxation_restores_constraints_and_keeps_earlier_solution(use_jit):
    distance = Parameter(0.03).variable(bounds=(0.01, 0.3))
    space = Space(d=distance)
    solver = Solver(System(space, source=Ray(x=0, theta=0.01, n=1)))
    solver.target(lambda sim: sim.final.x / 0.002 - 1)
    solver.require(Constraint(distance, upper_bound=0.15, label="mechanical clearance"))
    first = solver.solve(optimizer="scipy", jit=use_jit, x_upper=[0.1], options={"ftol": 1e-12})
    core = solver._cache.callbacks
    second = solver.solve(optimizer="scipy", jit=use_jit, x0=[0.08], x_upper=[0.3], options={"ftol": 1e-12})
    assert first.success and second.success and first.feasible and second.feasible
    assert solver._cache.callbacks is core
    assert first.value(distance) == pytest.approx(0.1, abs=1e-7)
    assert second.value(distance) == pytest.approx(0.15, abs=1e-7)
    assert first.simulation.parameter_data[0].upper_bound == 0.1
    assert second.simulation.parameter_data[0].upper_bound == 0.3
    assert solver.simulation.parameter_data[0].upper_bound == 0.3
    assert distance.value == 0.03
    assert any(entry.label == "mechanical clearance" for entry in first.constraints)
    assert first.run().final.x == pytest.approx(0.001, abs=1e-9)
    with pytest.raises(ValueError):
        first.x[0] = 2


def test_conflicting_runtime_constraints_return_unsuccessful_infeasible_solution():
    position = Parameter(0.5).variable(bounds=(-2, 2))
    solver = Solver(System(source=Ray(x=position, theta=0, n=1)))
    solver.target(position)
    # Result Symbols keep this contradiction outside affine preprocessing.
    solver.require(Constraint(lambda sim: sim.final.x, lower_bound=1, label="minimum"),
                   Constraint(lambda sim: -sim.final.x, lower_bound=1, label="maximum"))
    solution = solver.solve(optimizer="scipy", options={"maxiter": 15})
    assert not solution.success
    assert not solution.feasible
    assert solution.message
    assert solution.violations
    assert all(np.isfinite(solution.x))


def test_iteration_limit_is_distinct_from_feasibility():
    position = Parameter(0.5).variable(bounds=(-2, 2))
    solver = Solver(System(source=Ray(x=position, theta=0, n=1)))
    solver.target(lambda sim: sim.final.x ** 2 - 1)
    solution = solver.solve(optimizer="scipy", options={"maxiter": 1})
    assert solution.iterations == 1 and solution.function_evaluations >= 1
    assert not solution.success
    assert solution.feasible
    assert solution.message
    assert solution.violations == ()



def test_new_solver_from_solved_simulation_preserves_reduced_constraints():
    distance = Parameter(0.03).variable(bounds=(0.01, 0.3))
    solver = Solver(System(Space(d=distance), source=Ray(x=0, theta=0.01, n=1)))
    solver.target(lambda sim: sim.final.x / 0.002 - 1)
    solver.require(Constraint(distance, upper_bound=0.15, label="clearance"))
    first = solver.solve(optimizer="scipy", x_upper=[0.1], options={"ftol": 1e-12})
    assert solver._cache.output_indices == ()

    new_solver = Solver(first.simulation)
    new_solver.target(lambda sim: sim.final.x / 0.0025 - 1)
    second = new_solver.solve(optimizer="scipy", x_upper=[0.3], options={"ftol": 1e-12})
    assert first.success and first.feasible and second.success and second.feasible
    assert first.value(distance) == pytest.approx(0.1, abs=1e-7)
    assert second.value(distance) == pytest.approx(0.15, abs=1e-7)
    assert second.run().final.x == pytest.approx(0.0015, abs=1e-9)
    clearance = next(entry for entry in second.constraints if entry.label == "clearance")
    assert clearance.value == pytest.approx(0.15, abs=1e-7)
    assert clearance.upper_bound == 0.15
    assert first.simulation.parameter_data[0].upper_bound == 0.1
    assert first.simulation.initial_values[distance] == pytest.approx(0.1, abs=1e-7)
    assert solver.simulation.initial_values[distance] == 0.03
    assert solver.simulation.parameter_data[0].upper_bound == 0.3
    assert distance.value == 0.03


def test_native_scipy_restores_partially_fixed_canonical_coordinates():
    position = Parameter(0.2).variable(bounds=(-1, 1))
    angle = Parameter(0.1).variable(bounds=(-1, 1))
    solver = Solver(System(Space(d=1), source=Ray(x=position, theta=angle, n=1)))
    solver.target(lambda sim: sim.final.x - 1)
    indices = solver.simulation._optics_program.parameter_indices
    lower, upper = np.full(2, -1.), np.full(2, 1.)
    lower[indices[id(position)]] = upper[indices[id(position)]] = 0.2
    solution = solver.solve(optimizer="scipy", x_lower=lower, x_upper=upper, options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(position) == pytest.approx(0.2, abs=1e-12)
    assert solution.value(angle) == pytest.approx(0.8, abs=1e-7)
    assert solution.run().final.x == pytest.approx(1, abs=1e-7)
    assert solution.cost < 1e-12
    np.testing.assert_allclose(solution.simulation.initial_values.values, solution.x)
    info = solution.simulation.parameter_data[indices[id(position)]]
    assert info.lower_bound == info.upper_bound == 0.2
    assert solver.simulation.initial_values[position] == position.value == 0.2
    assert solver.simulation.initial_values[angle] == angle.value == 0.1
    assert all(item.lower_bound == -1 and item.upper_bound == 1
               for item in solver.simulation.parameter_data)


def test_value_reads_derived_design_quantities_after_a_real_solve():
    """A gap defined from the remaining length is not a Parameter; value() evaluates it."""
    system_length = 0.15
    distance = Parameter(0.05).variable(bounds=(0.02, 0.12))
    first = Space(d=distance, label="First gap")
    remainder = Space(d=system_length - distance, label="Remainder")
    system = System(first, remainder, source=Ray(x=0, theta=0.01, n=1))
    system.bind("double", 2 * distance)
    solver = Solver(system).target(lambda sim: sim[first].z_end / 0.08 - 1)
    solution = solver.solve(optimizer="scipy", options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    result = solution.run()

    assert solution.value(distance) == pytest.approx(0.08, abs=1e-7)
    assert solution.value(remainder.d) == pytest.approx(0.07, abs=1e-7)
    assert solution.value(remainder.d) == pytest.approx(result[remainder].z_end - result[remainder].z_start)
    assert solution.value(first.d + remainder.d) == pytest.approx(system_length)
    assert solution.value(Symbol("double") / 4) == pytest.approx(0.04, abs=1e-7)
    # The declaration is untouched.
    assert distance.value == 0.05 and remainder.d.value == pytest.approx(0.10)
