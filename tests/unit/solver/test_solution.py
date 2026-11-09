"""Solution reporting uses complete compiled models in physical coordinates."""

from types import SimpleNamespace

import numpy as np
import pytest

from diffractix import Constraint, Objective, Parameter, Ray, Solver, System
from diffractix.backend import xp
from diffractix.solver import Solution, SolverResult, DEFAULT_FEASIBILITY_TOLERANCE
from diffractix.solver.solution import ParameterBoundResult


def make_solution(*, solver=None, x=(2.,), lower=None, upper=None,
                  success=True, tolerance=DEFAULT_FEASIBILITY_TOLERANCE):
    if solver is None:
        solver = Solver(System(optical_input=Ray(x=Parameter(0.5).variable(), theta=0, n=1)))
    cache = solver._build_cache(False, solver.simulation.initial_values.values)
    result = SolverResult(np.array(x), success, 7.5, "stopped")
    problem = SimpleNamespace(
        x_lower=np.full(len(solver.simulation.parameter_data), -np.inf) if lower is None else np.array(lower),
        x_upper=np.full(len(solver.simulation.parameter_data), np.inf) if upper is None else np.array(upper),
    )
    solution = Solution(
        result, problem, solver.simulation,
        objective_program=cache.objective_program,
        objective_descriptors=cache.objective_descriptors,
        callable_objectives=cache.callable_objectives,
        constraint_program=cache.constraint_program,
        constraint_descriptors=cache.constraint_descriptors,
        callable_constraints=cache.callable_constraints,
        feasibility_tolerance=tolerance,
    )
    return solution, result


def test_backend_status_and_owned_x():
    solution, result = make_solution(success=False)
    result.x[0] = 100
    np.testing.assert_array_equal(solution.x, [2])
    with pytest.raises(ValueError):
        solution.x[0] = 3
    assert not solution.success
    assert solution.cost == 7.5
    assert solution.message == "stopped"
    assert solution.feasible


@pytest.mark.parametrize("x", [(), (1., 2.), ((1.,),), 1.])
def test_invalid_result_shape(x):
    with pytest.raises(ValueError, match="x shape"):
        make_solution(x=x)


@pytest.mark.parametrize("x", [np.nan, np.inf, -np.inf])
def test_nonfinite_result_x(x):
    with pytest.raises(ValueError, match="finite"):
        make_solution(x=(x,))


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf, -np.inf])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite.*non-negative"):
        make_solution(tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [True, 1j, "0", object()])
def test_wrong_tolerance_type(tolerance):
    with pytest.raises(TypeError, match="real"):
        make_solution(tolerance=tolerance)


def test_updated_simulation_and_canonical_value_lookup():
    p = Parameter(0.5).variable()
    source = Ray(x=p, theta=0, n=1)
    solver = Solver(System(optical_input=source))
    solution, _ = make_solution(solver=solver, x=(2.,), lower=[1.], upper=[3.])
    assert solution.simulation is not solver.simulation
    assert solution.value(p) == solution.value(source.x) == 2
    np.testing.assert_array_equal(solution.simulation.initial_values.values, [2])
    assert solution.simulation.parameter_data[0].lower_bound == 1
    assert solution.simulation.parameter_data[0].upper_bound == 3
    assert solver.simulation.parameter_data[0].value == 0.5
    assert p.value == 0.5
    p.value = 20
    assert solution.run().final.x == pytest.approx(2)
    assert solution.simulation.run().final.x == pytest.approx(2)
    with pytest.raises(KeyError):
        solution.value(Parameter(1).variable())
    with pytest.raises(TypeError):
        solution.value(p + 1)


def test_complete_evaluations_preserve_interleaved_order_and_vector_shapes():
    p = Parameter(0.5).variable()
    system = System(optical_input=Ray(x=p, theta=0, n=1))
    system.require(Constraint(p, lower_bound=0, label="system"))
    solver = Solver(system)

    def dynamic(result):
        if result.final.x > 0:
            return xp.array([result.final.x, -result.final.x])
        return xp.zeros(2)

    solver.target(Objective(dynamic, weight=3, label="dynamic"),
                  Objective(p - 1, weight=2, label="parameter"),
                  Objective(lambda result: result.final.x, label="runtime"))
    solver.require(Constraint(dynamic, lower_bound=-1, upper_bound=1, label="vector"),
                   Constraint(p ** 2, upper_bound=3, label="square"))
    solution, _ = make_solution(solver=solver)
    assert [entry.label for entry in solution.objectives] == ["dynamic", "parameter", "runtime"]
    np.testing.assert_allclose([entry.cost for entry in solution.objectives], [24, 2, 4])
    assert [entry.position for entry in solution.objectives] == [0, 1, 2]
    assert [entry.label for entry in solution.constraints] == ["system", "vector", "square"]
    vector = solution.constraints[1]
    np.testing.assert_allclose(vector.value, [2, -2])
    np.testing.assert_allclose(vector.violation, [1, 1])
    assert not vector.satisfied
    assert [entry.label for entry in solution.violations] == ["vector", "square"]
    assert not solution.feasible
    assert solution.run() is solution.run()


@pytest.mark.parametrize("lower,upper,x,margin", [
    (1, np.inf, 2, 1), (1, np.inf, 0.5, -0.5),
    (-np.inf, 3, 2, 1), (-np.inf, 3, 4, -1),
    (1, 3, 2, 1), (1, 3, 4, -1), (2, 2, 2, 0), (2, 2, 2.25, -0.25),
])
def test_constraint_margins_and_violations(lower, upper, x, margin):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.require(Constraint(p, lower_bound=lower, upper_bound=upper))
    solution, _ = make_solution(solver=solver, x=(x,))
    entry = solution.constraints[0]
    assert entry.margin == pytest.approx(margin)
    assert entry.violation == pytest.approx(max(-margin, 0))
    assert entry.satisfied == (margin >= 0)


@pytest.mark.parametrize("bound,direction", [(0., -1), (1., 1)])
@pytest.mark.parametrize("tolerance", [0., 0.125])
def test_bound_violation_reporting_respects_tolerance(bound, direction, tolerance):
    edge = bound + direction * tolerance
    on_edge, _ = make_solution(x=(edge,), lower=[0], upper=[1], tolerance=tolerance)
    outside, _ = make_solution(x=(np.nextafter(edge, direction * np.inf),),
                               lower=[0], upper=[1], tolerance=tolerance)
    assert on_edge.feasible
    assert not outside.feasible
    assert outside.success
    assert isinstance(outside.violations[0], ParameterBoundResult)
    assert outside.violations[0].violation > tolerance


def test_reporting_shares_one_optical_result(monkeypatch):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.target(lambda result: result.final.x)
    solver.require(Constraint(lambda result: result.final.x, upper_bound=1))
    solution, _ = make_solution(solver=solver)
    calls = []
    run = solution.simulation.run

    def record(theta):
        calls.append(theta)
        return run(theta)

    monkeypatch.setattr(solution.simulation, "run", record)
    solution.objectives
    solution.constraints
    solution.violations
    solution.run()
    assert len(calls) == 1


def test_solve_reports_reduced_constraints_and_keeps_overrides_local(monkeypatch):
    from diffractix.solver import solver as solver_module

    def backend(problem, **kwargs):
        physical = min(3., float(problem.scaling.x_center[0] + problem.scaling.x_scale[0] * problem.x_upper[0]))
        numerical = (np.array([physical]) - problem.scaling.x_center) / problem.scaling.x_scale
        return SolverResult(numerical, True, (physical - 3) ** 2, "controlled result")

    monkeypatch.setattr(solver_module, "solve_scipy", backend)
    p = Parameter(0.5).variable(bounds=(0, 5))
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.target(p - 3)
    solver.require(Constraint(p, lower_bound=1, label="bound-redundant"))
    solution = solver.solve(x0=[1.5], x_lower=[1], x_upper=[2])
    assert solution.success and solution.feasible
    assert solution.x[0] == pytest.approx(2, abs=1e-6)
    assert solution.cost == pytest.approx(1, abs=1e-6)
    assert len(solution.constraints) == 1
    assert solution.constraints[0].label == "bound-redundant"
    assert solver._cache.output_indices == ()
    assert solution.simulation.parameter_data[0].upper_bound == 2
    assert solver.simulation.parameter_data[0].upper_bound == 5
    second = solver.solve(x_upper=[4])
    assert second.x[0] == pytest.approx(3, abs=1e-6)
    assert solution.x[0] == pytest.approx(2, abs=1e-6)
    assert solution.constraints[0].value == pytest.approx(2, abs=1e-6)


@pytest.mark.parametrize("value,satisfied", [(2.125, True), (2.25, False)])
def test_constraint_reporting_tolerance(value, satisfied):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.require(Constraint(p, lower_bound=2, upper_bound=2))
    solution, _ = make_solution(solver=solver, x=(value,), tolerance=0.125)
    assert solution.constraints[0].satisfied is satisfied
    assert solution.feasible is satisfied


def test_empty_reporting_and_value_lookup_need_no_propagation(monkeypatch):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solution, _ = make_solution(solver=solver)

    def unexpected_run(*args):
        raise AssertionError("Reporting constants and theta does not need propagation")

    monkeypatch.setattr(solution.simulation, "run", unexpected_run)
    assert solution.value(p) == 2
    assert solution.objectives == solution.constraints == solution.violations == ()
    assert solution.feasible


def test_parameter_bound_checks_cover_all_canonical_coordinates():
    first = Parameter(0.5).variable()
    second = Parameter(0.25).variable()
    solver = Solver(System(optical_input=Ray(x=first, theta=second, n=1)))
    solution, _ = make_solution(solver=solver, x=(2., -3.),
                                lower=[-np.inf, 0], upper=[np.inf, 1])
    assert solution.value(first) == 2
    assert solution.value(second) == -3
    assert len(solution.violations) == 1
    assert solution.violations[0].parameter.parameter_id == id(second)
    assert solution.violations[0].violation == 3


@pytest.mark.parametrize("kind", ["objective", "constraint"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_reporting_is_rejected(kind, invalid):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))

    def residual(result):
        if result.final.x > 1:
            return invalid
        return result.final.x

    if kind == "objective":
        solver.target(residual, label="bad")
    else:
        solver.require(Constraint(residual, label="bad"))
    solution, _ = make_solution(solver=solver)
    with pytest.raises(ValueError, match="bad.*non-finite"):
        getattr(solution, "objectives" if kind == "objective" else "constraints")
