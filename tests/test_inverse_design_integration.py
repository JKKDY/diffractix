from __future__ import annotations

import autograd.numpy as np
import numpy as numpy
import pytest

from diffractix.beams import GaussianBeam
from diffractix.elements import Space
from diffractix.graph import Parameter, Symbol
from diffractix.solver import (
    Backend,
    Constraint,
    Objective,
    OptimizationResult,
    Solution,
    Solver,
)
from diffractix.solver.context import SolverSymbolKey, SolverSymbolKind
from diffractix.solver.solution import DEFAULT_FEASIBILITY_TOLERANCE
from diffractix.system import System


def beam():
    return GaussianBeam.from_waist(
        w0=100e-6,
        wavelength=1064e-9,
    )


def distance_system(
    initial=0.1,
    *,
    lower=0.02,
    upper=0.4,
):
    distance = Parameter(
        initial,
        name="distance",
        variable=True,
        lower_bound=lower,
        upper_bound=upper,
    )
    space = Space(d=distance, label="design space")
    system = System()
    system.add_input_beam(beam())
    system.add(space)
    return system, space, distance


def solve_distance(system, target=0.2, **kwargs):
    solver = Solver(system)
    solver.target(lambda context: context.z[1] - target)
    return solver, solver.solve(Backend.SCIPY, method="SLSQP", **kwargs)


def test_real_optical_inverse_design_public_workflow():
    system, space, distance = distance_system()
    initial_value = distance.value
    solver, solution = solve_distance(system)

    assert isinstance(solution, Solution)
    assert solution.success
    assert solution.feasible
    numpy.testing.assert_allclose(solution.x, [0.2], atol=1e-6)
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)
    assert solution.run().z[-1] == pytest.approx(0.2, abs=1e-6)
    assert solution.targets[0].value == pytest.approx(0.0, abs=1e-6)
    assert distance.value == initial_value
    assert distance.is_variable
    assert solver.simulation.run().z[-1] == pytest.approx(initial_value)


def test_multiple_real_design_variables_preserve_canonical_theta_order():
    first = Parameter(0.08, name="first", variable=True, lower_bound=0.02, upper_bound=0.3)
    second = Parameter(0.12, name="second", variable=True, lower_bound=0.02, upper_bound=0.3)
    first_space = Space(d=first)
    second_space = Space(d=second)
    system = System()
    system.add_input_beam(beam())
    system.add((first_space, second_space))
    solver = Solver(system)
    solver.target(
        lambda context: context.z[1] - 0.1,
        lambda context: context.z[2] - 0.3,
    )

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert len(solution.x) == 2
    for index, info in enumerate(solution.parameter_info):
        assert info.parameter_index == index
        assert solution.parameters[index].info is info
    first_index = solver.simulation.parameter_info[id(first)].parameter_index
    second_index = solver.simulation.parameter_info[id(second)].parameter_index
    assert solution[first] == pytest.approx(solution.x[first_index])
    assert solution[second] == pytest.approx(solution.x[second_index])
    numpy.testing.assert_allclose(
        [solution[first], solution[second]],
        [0.1, 0.2],
        atol=1e-5,
    )


def test_shared_parameter_and_dependent_expressions_remain_canonical():
    base = Parameter(0.08, name="base", variable=True, lower_bound=0.02, upper_bound=0.2)
    doubled = 2.0 * base
    shifted = doubled + 3.0
    first = Space(d=base)
    second = Space(d=doubled)
    system = System()
    system.add_input_beam(beam())
    system.add((first, second))
    solver = Solver(system)
    solver.target(lambda context: context.z[2] - 0.3)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert len(solution.x) == 1
    assert solution[base] == pytest.approx(0.1, abs=1e-6)
    assert solution[doubled] == pytest.approx(2.0 * solution[base])
    assert solution[shifted] == pytest.approx(2.0 * solution[base] + 3.0)
    assert base.value == 0.08


def test_real_parameter_bound_becomes_active_and_is_reported():
    system, _, distance = distance_system(upper=0.3)
    _, solution = solve_distance(system, target=0.5)

    assert solution.success
    assert solution[distance] == pytest.approx(0.3, abs=1e-6)
    assert solution.parameters[0].info.lower_bound == pytest.approx(0.02)
    assert solution.parameters[0].info.upper_bound == pytest.approx(0.3)


def test_real_hard_constraint_changes_the_optimum():
    unconstrained_system, _, unconstrained_distance = distance_system(initial=0.15)
    unconstrained = Solver(unconstrained_system)
    unconstrained.target(unconstrained_distance - 0.05)
    unconstrained_solution = unconstrained.solve(Backend.SCIPY, method="SLSQP")

    constrained_system, _, constrained_distance = distance_system(initial=0.15)
    constrained = Solver(constrained_system)
    constrained.target(constrained_distance - 0.05)
    constrained.require(constrained_distance >= 0.2)
    constrained_solution = constrained.solve(Backend.SCIPY, method="SLSQP")

    assert unconstrained_solution[unconstrained_distance] == pytest.approx(0.05, abs=1e-6)
    assert constrained_solution[constrained_distance] == pytest.approx(0.2, abs=1e-6)
    assert constrained_solution.success
    assert constrained_solution.feasible
    assert constrained_solution.violations == ()
    assert constrained_solution.constraints[0].satisfied


def test_multiple_constraint_kinds_are_preserved_in_solution():
    system, _, distance = distance_system(initial=0.18)
    solver = Solver(system)
    solver.target(distance - 0.3)
    solver.require(
        distance >= 0.1,
        distance <= 0.3,
        Constraint(distance, lower_bound=0.15, upper_bound=0.25),
        distance == 0.2,
    )

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    assert len(solution.constraints) == 4
    assert all(result.satisfied for result in solution.constraints)
    assert solution.violations == ()
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)


def test_weighted_objectives_report_the_backend_cost():
    system, _, distance = distance_system(initial=0.2)
    first = Objective(distance - 0.1, weight=1.0, label="near")
    second = Objective(distance - 0.3, weight=4.0, label="far")
    solver = Solver(system)
    solver.target(first, second)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert tuple(target.weight for target in solution.targets) == (1.0, 4.0)
    assert tuple(target.objective.label for target in solution.targets) == ("near", "far")
    assert sum(target.cost for target in solution.targets) == pytest.approx(solution.cost)


def test_real_runtime_lookup_sources_reproduce_solution_run():
    system, space, _ = distance_system()
    reference = system.build().run(np.array([0.2]))
    desired_w = reference.after(space).w
    solver = Solver(system)
    solver.target(
        lambda context: context.z[1] - 0.2,
        lambda context: context.at(space).w - reference.at(space).w,
        lambda context: context.after(space).w - desired_w,
        lambda context: context.states[1].w - desired_w,
    )

    solution = solver.solve(Backend.SCIPY, method="SLSQP")
    result = solution.run()

    assert solution.success
    expected = (
        result.z[1] - 0.2,
        result.at(space).w - reference.at(space).w,
        result.after(space).w - desired_w,
        result.states[1].w - desired_w,
    )
    numpy.testing.assert_allclose(
        [target.value for target in solution.targets],
        expected,
        atol=1e-8,
    )


def test_dynamic_python_control_flow_survives_real_solve():
    system, _, distance = distance_system(initial=0.1)
    solver = Solver(system)

    def target(context):
        if context.z[1] > 0.15:
            return distance - 0.2
        return 2.0 * (distance - 0.2)

    solver.target(target)
    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)
    assert solution.targets[0].value == pytest.approx(0.0, abs=1e-6)
    assert distance.value == 0.1


def test_raw_truth_and_boolean_operators_use_dynamic_fallback():
    system, _, distance = distance_system(initial=0.1)
    solver = Solver(system)

    def raw_and_or(context):
        if context.z[1] and (context.z[1] > 0.0 or context.z[1] < -1.0):
            return distance - 0.2
        return distance + 1.0

    def symbolic_not(context):
        if not context.z[1]:
            return distance + 1.0
        return distance - 0.2

    solver.target(raw_and_or, symbolic_not)
    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)
    numpy.testing.assert_allclose(
        [target.value for target in solution.targets],
        [0.0, 0.0],
        atol=1e-6,
    )


def test_unsupported_symbolic_conversion_propagates_original_error():
    system, _, _ = distance_system()
    solver = Solver(system)
    solver.target(lambda context: float(context.z[1]))

    with pytest.raises(RuntimeError, match="has no standalone value"):
        solver.solve(Backend.SCIPY, method="SLSQP")


def test_repeated_element_occurrences_resolve_independently():
    distance = Parameter(0.1, variable=True, lower_bound=0.02, upper_bound=0.3)
    repeated = Space(d=distance)
    system = System()
    system.add_input_beam(beam())
    system.add(repeated)
    system.add(repeated)
    solver = Solver(system)
    solver.target(lambda context: context.z[2] - 0.3)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")
    result = solution.run()

    assert solution.success
    assert result.at(repeated, occurrence=0) is result.states[0]
    assert result.at(repeated, occurrence=1) is result.states[1]
    assert result.at(repeated, occurrence=0).w != pytest.approx(
        result.at(repeated, occurrence=1).w
    )


def test_solver_expression_resolves_explicit_element_occurrence():
    distance = Parameter(
        0.1,
        variable=True,
        lower_bound=0.02,
        upper_bound=0.3,
    )
    repeated = Space(d=distance)
    system = System()
    system.add_input_beam(beam())
    system.add(repeated)
    system.add(repeated)
    simulation = system.build()
    desired_distance = 0.15
    desired_w = simulation.run(np.array([desired_distance])).at(
        repeated,
        occurrence=1,
    ).w
    solver = Solver(simulation)
    solver.target(
        lambda context: 1e4 * (
            context.at(repeated, occurrence=1).w - desired_w
        )
    )

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution[distance] == pytest.approx(desired_distance, abs=1e-6)
    assert solution.run().at(repeated, occurrence=1).w == pytest.approx(
        desired_w,
        abs=1e-10,
    )


def test_reusing_built_simulation_produces_independent_solutions():
    system, _, distance = distance_system()
    simulation = system.build()
    first = Solver(simulation)
    second = Solver(simulation)
    first.target(distance - 0.15)
    second.target(distance - 0.25)

    first_solution = first.solve(Backend.SCIPY, method="SLSQP")
    second_solution = second.solve(Backend.SCIPY, method="SLSQP")

    assert first_solution[distance] == pytest.approx(0.15, abs=1e-6)
    assert second_solution[distance] == pytest.approx(0.25, abs=1e-6)
    assert distance.value == 0.1
    numpy.testing.assert_array_equal(simulation.initial_values, [0.1])


def test_multiple_solves_start_from_frozen_simulation_values():
    system, _, distance = distance_system()
    solver = Solver(system)
    solver.target(distance - 0.15)
    first = solver.solve(Backend.SCIPY, method="SLSQP", options={"maxiter": 100})
    second = solver.solve(Backend.SCIPY, method="SLSQP", options={"maxiter": 20})

    assert first is not second
    assert first[distance] == pytest.approx(0.15, abs=1e-6)
    assert second[distance] == pytest.approx(0.15, abs=1e-6)
    assert distance.value == 0.1


def test_solver_accepts_system_and_prebuilt_simulation():
    first_system, _, first_distance = distance_system()
    second_system, _, second_distance = distance_system()
    from_system = Solver(first_system)
    from_simulation = Solver(second_system.build())
    from_system.target(first_distance - 0.2)
    from_simulation.target(second_distance - 0.2)

    first = from_system.solve(Backend.SCIPY, method="SLSQP")
    second = from_simulation.solve(Backend.SCIPY, method="SLSQP")

    numpy.testing.assert_allclose(first.x, second.x, atol=1e-6)


def test_prebuilt_simulation_uses_frozen_parameter_snapshot():
    system, _, distance = distance_system()
    simulation = system.build()
    distance.value = 0.3
    distance.fixed()
    solver = Solver(simulation)
    solver.target(distance - 0.2)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert len(solution.x) == 1
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)
    assert distance.value == 0.3
    assert distance.is_variable is False


def test_replacing_element_parameter_after_build_does_not_enter_frozen_simulation():
    system, space, original = distance_system()
    simulation = system.build()
    replacement = Parameter(
        0.25,
        name="replacement",
        variable=True,
        lower_bound=0.02,
        upper_bound=0.4,
    )
    space.d = replacement
    solver = Solver(simulation)
    solver.target(space.d - 0.2)

    with pytest.raises(ValueError, match="not part of this Simulation"):
        solver.solve(Backend.SCIPY, method="SLSQP")

    assert id(original) in simulation.parameter_info
    assert id(replacement) not in simulation.parameter_info
    assert simulation.run().z[-1] == pytest.approx(0.1)


def test_solution_is_stable_after_solver_mutation():
    system, _, distance = distance_system()
    solver, solution = solve_distance(system)
    original_targets = solution.targets
    original_constraints = solution.constraints
    original_cost = solution.cost
    original_feasible = solution.feasible

    solver.target(distance - 0.35)
    solver.require(distance >= 0.3)

    assert solution.targets is original_targets
    assert solution.constraints is original_constraints
    assert solution.cost == original_cost
    assert solution.feasible is original_feasible


def test_cached_solution_runtime_result_is_stable():
    system, _, _ = distance_system()
    _, solution = solve_distance(system)

    first_run = solution.run()
    first_targets = solution.targets
    first_constraints = solution.constraints

    assert solution.run() is first_run
    assert solution.targets is first_targets
    assert solution.constraints is first_constraints


def test_infeasible_problem_returns_non_feasible_solution():
    system, _, distance = distance_system(initial=0.15)
    solver = Solver(system)
    solver.target(distance - 0.15)
    solver.require(distance >= 0.25, distance <= 0.1)

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.feasible is False
    assert solution.violations
    assert solution.success is False


@pytest.mark.parametrize(
    ("tolerance", "expected"),
    [(0.0, False), (1e-3, True), (np.inf, True)],
)
def test_solution_reporting_tolerance_is_independent_of_backend_options(
    monkeypatch,
    tolerance,
    expected,
):
    from diffractix.solver import solver as solver_module

    system, _, distance = distance_system(initial=0.2)
    solver = Solver(system)
    solver.target(distance - 0.2)
    solver.require(distance >= 0.2005)

    def solve_scipy(problem, method=None, options=None):
        return OptimizationResult(
            x=np.array([0.2]),
            success=True,
            cost=0.0,
            message="mocked reporting point",
        )

    monkeypatch.setattr(solver_module, "solve_scipy", solve_scipy)

    solution = solver.solve(
        Backend.SCIPY,
        method="SLSQP",
        options={"ftol": 1e-9},
        feasibility_tolerance=tolerance,
    )

    assert solution.feasibility_tolerance == tolerance
    assert solution.constraints[0].satisfied is expected


def test_invalid_backend_fails_clearly_before_dispatch():
    system, _, _ = distance_system()
    solver = Solver(system)

    with pytest.raises(ValueError, match="Unsupported backend"):
        solver.solve("definitely-not-a-backend")


def test_invalid_scipy_method_and_option_propagate():
    from scipy.optimize import OptimizeWarning

    system, _, distance = distance_system()
    solver = Solver(system)
    solver.target(distance - 0.2)

    with pytest.raises(ValueError, match="Unsupported SciPy method"):
        solver.solve(Backend.SCIPY, method="DOES_NOT_EXIST")

    with pytest.warns(OptimizeWarning, match="Unknown solver options"):
        solver.solve(
            Backend.SCIPY,
            method="SLSQP",
            options={"definitely_not_an_option": 1},
        )


def test_negative_feasibility_tolerance_fails_cleanly():
    system, _, distance = distance_system()
    solver = Solver(system)
    solver.target(distance - 0.2)

    with pytest.raises(ValueError, match="non-negative"):
        solver.solve(
            Backend.SCIPY,
            method="SLSQP",
            feasibility_tolerance=-1.0,
        )


def test_unknown_runtime_element_error_propagates():
    system, _, _ = distance_system()
    missing = Space(d=0.1)
    solver = Solver(system)
    solver.target(lambda context: context.at(missing).w)

    with pytest.raises(KeyError, match="not part of this simulation"):
        solver.solve(Backend.SCIPY, method="SLSQP")


def test_invalid_runtime_indices_propagate():
    system, space, _ = distance_system()

    bad_occurrence = Solver(system)
    bad_occurrence.target(lambda context: context.at(space, occurrence=3).w)
    with pytest.raises(IndexError, match="out of range"):
        bad_occurrence.solve(Backend.SCIPY, method="SLSQP")

    bad_state = Solver(system.build())
    bad_state.target(lambda context: context.states[99].w)
    with pytest.raises(IndexError):
        bad_state.solve(Backend.SCIPY, method="SLSQP")


def test_invalid_z_index_and_negative_occurrence_propagate():
    system, space, _ = distance_system()

    bad_z = Solver(system)
    bad_z.target(lambda context: context.z[99])
    with pytest.raises(IndexError):
        bad_z.solve(Backend.SCIPY, method="SLSQP")

    negative_occurrence = Solver(system.build())
    negative_occurrence.target(
        lambda context: context.at(space, occurrence=-1).w
    )
    with pytest.raises(IndexError, match="out of range"):
        negative_occurrence.solve(Backend.SCIPY, method="SLSQP")


def test_direct_runtime_symbol_is_rejected_before_backend():
    system, _, _ = distance_system()
    solver = Solver(system)
    solver.target(Symbol(SolverSymbolKey(SolverSymbolKind.Z, index=0)))

    with pytest.raises(ValueError, match="no compile context"):
        solver.solve(Backend.SCIPY, method="SLSQP")


def test_external_variable_is_rejected_but_external_fixed_is_constant():
    system, _, distance = distance_system()
    external_variable = Parameter(0.3, name="external").variable()
    invalid = Solver(system)
    invalid.target(distance - external_variable)

    with pytest.raises(ValueError, match="not part of this Simulation"):
        invalid.solve(Backend.SCIPY, method="SLSQP")

    external_fixed = Parameter(0.25, name="external fixed")
    valid = Solver(system.build())
    valid.target(distance - external_fixed)
    solution = valid.solve(Backend.SCIPY, method="SLSQP")
    assert solution[distance] == pytest.approx(0.25, abs=1e-6)


def test_user_exception_is_not_swallowed_and_retry_is_clean():
    system, _, distance = distance_system()
    solver = Solver(system)

    def broken(context):
        raise RuntimeError("user failure")

    solver.target(broken)
    with pytest.raises(RuntimeError, match="user failure"):
        solver.solve(Backend.SCIPY, method="SLSQP")

    solver.objectives.clear()
    solver.target(distance - 0.2)
    solution = solver.solve(Backend.SCIPY, method="SLSQP")
    assert solution.success
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)


def test_constraint_exception_is_not_swallowed_and_retry_is_clean():
    system, _, distance = distance_system()
    solver = Solver(system)

    def broken(context):
        raise RuntimeError("constraint failure")

    solver.require(Constraint(broken, lower_bound=0.0))
    with pytest.raises(RuntimeError, match="constraint failure"):
        solver.solve(Backend.SCIPY, method="SLSQP")

    solver.constraints.clear()
    solver.target(distance - 0.2)
    solver.require(distance >= 0.05)
    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert solution.success
    assert solution.feasible
    assert solution[distance] == pytest.approx(0.2, abs=1e-6)


def test_backend_exception_does_not_poison_solver_retry(monkeypatch):
    from diffractix.solver import solver as solver_module

    system, _, distance = distance_system()
    solver = Solver(system)
    solver.target(distance - 0.2)
    calls = 0

    def flaky_backend(problem, method=None, options=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("backend failure")
        return OptimizationResult(
            x=np.array([0.2]),
            success=True,
            cost=problem.objective(np.array([0.2])),
            message="recovered",
        )

    monkeypatch.setattr(solver_module, "solve_scipy", flaky_backend)

    with pytest.raises(RuntimeError, match="backend failure"):
        solver.solve(Backend.SCIPY, method="SLSQP")

    solution = solver.solve(Backend.SCIPY, method="SLSQP")

    assert calls == 2
    assert solution.success
    assert solution.message == "recovered"
    assert solution[distance] == pytest.approx(0.2)


@pytest.mark.xfail(
    reason="Solver should define or clearly reject an optimization with no objectives or constraints.",
    strict=False,
)
def test_empty_optimization_problem_should_have_a_clear_contract():
    system, _, _ = distance_system()
    solver = Solver(system)

    with pytest.raises(ValueError, match="objective|constraint|empty"):
        solver.solve(Backend.SCIPY, method="SLSQP")


@pytest.mark.xfail(
    reason="Solver-level validation should reject zero-variable problems before backend dispatch.",
    strict=False,
)
def test_no_variables_should_raise_clear_solver_error():
    system = System()
    system.add_input_beam(beam())
    system.add(Space(d=0.1))
    solver = Solver(system)
    solver.target(lambda context: 1.0)

    with pytest.raises(ValueError, match="variable"):
        solver.solve(Backend.SCIPY, method="SLSQP")


@pytest.mark.xfail(
    reason="Future validation should reject dynamic constraint shape changes before optimization.",
    strict=False,
)
def test_dynamic_constraint_shape_change_should_be_rejected_early():
    system, _, distance = distance_system(initial=0.1)
    solver = Solver(system)

    def changing_shape(context):
        if context.z[1] > 0.15:
            return np.array([distance, distance])
        return distance

    solver.require(Constraint(changing_shape, lower_bound=0.0))

    with pytest.raises(ValueError, match="shape"):
        solver.solve(Backend.SCIPY, method="SLSQP")
