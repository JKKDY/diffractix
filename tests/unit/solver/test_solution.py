"""Solution reporting uses complete compiled models in physical coordinates."""

from types import SimpleNamespace

import numpy as np
import pytest

from diffractix import Constraint, Objective, Parameter, Ray, Solver, Space, System
from diffractix.graph import InputNode, Literal, Symbol, UnresolvedInputError
from diffractix.simulation.result import ResultSymbolKey, ResultSymbolKind
from diffractix.backend import xp
from diffractix.solver import Solution, SolverResult, DEFAULT_FEASIBILITY_TOLERANCE
from diffractix.solver.solution import ParameterBoundResult


def make_solution(*, solver=None, x=(2.,), lower=None, upper=None,
                  success=True, tolerance=DEFAULT_FEASIBILITY_TOLERANCE,
                  iterations=None, function_evaluations=None):
    if solver is None:
        solver = Solver(System(optical_input=Ray(x=Parameter(0.5).variable(), theta=0, n=1)))
    cache = solver._build_cache(False, solver.simulation.initial_values.values)
    result = SolverResult(np.array(x), success, 7.5, "stopped",
                          iterations=iterations, function_evaluations=function_evaluations)
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


def test_backend_performance_counts_are_reported_or_unknown():
    reported, _ = make_solution(iterations=11, function_evaluations=14)
    assert (reported.iterations, reported.function_evaluations) == (11, 14)
    unknown, _ = make_solution()
    assert unknown.iterations is None and unknown.function_evaluations is None


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
    with pytest.raises(TypeError, match="Parameter or graph expression"):
        solution.value(1.0)


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


def two_parameter_solution(**bindings):
    """Solution with first=2 and second=-3, whatever the canonical order of the two."""
    first, second = Parameter(0.5).variable(), Parameter(0.25).variable()
    system = System(optical_input=Ray(x=first, theta=second, n=1))
    for key, build in bindings.items():
        system.bind(key, build(first, second))
    solver = Solver(system)
    indices = solver.simulation._optics_program.parameter_indices
    x = np.zeros(2)
    x[indices[id(first)]], x[indices[id(second)]] = 2., -3.
    solution, _ = make_solution(solver=solver, x=tuple(x))
    return solution, first, second


def test_value_evaluates_expressions_of_the_solved_parameters():
    solution, first, second = two_parameter_solution()
    assert solution.value(first - 2 * second) == pytest.approx(8.)
    assert solution.value(second - 2 * first) == pytest.approx(-7.)
    assert solution.value((first + 1) * (second - 1) / 4 + 0.5) == pytest.approx(-2.5)
    assert solution.value(first.sin() ** 2 + first.cos() ** 2) == pytest.approx(1.)
    assert solution.value(InputNode(first * 3)) == pytest.approx(6.)
    assert solution.value(InputNode(InputNode(first))) == pytest.approx(2.)


def test_value_expression_results_are_plain_numbers_and_arrays():
    solution, first, _ = two_parameter_solution()
    assert type(solution.value(first * 2)) is float
    assert type(solution.value(first * (1 + 1j))) is complex
    array = solution.value(first * Literal(np.array([1., 2.])))
    assert isinstance(array, np.ndarray) and array.shape == (2,)
    np.testing.assert_allclose(array, [2., 4.])


def test_value_resolves_symbols_through_the_simulation_bindings():
    solution, first, second = two_parameter_solution(
        twice=lambda a, b: 2 * a, alias=lambda a, b: Symbol("twice") + b, constant=lambda a, b: 5.0,
    )
    assert solution.value(Symbol("twice") + 1) == pytest.approx(5.)
    assert solution.value(Symbol("alias") * 2) == pytest.approx(2 * (4. - 3.))
    assert solution.value(Symbol("constant") * first) == pytest.approx(10.)


def test_value_uses_solved_values_not_live_parameter_state():
    solution, first, _ = two_parameter_solution()
    first.value = 99.
    first.fixed()
    assert solution.value(first) == 2.
    assert solution.value(first * 10) == pytest.approx(20.)


def test_value_uses_the_current_value_of_fixed_parameters():
    solution, first, _ = two_parameter_solution(constant=lambda a, b: 5.0)
    fixed = Parameter(3.0, name="fixed")
    assert solution.value(fixed * first) == pytest.approx(6.)
    assert solution.value(Symbol("constant") * first) == pytest.approx(10.)
    fixed.value = 4.0   # documented: fixed Parameters are read live, like in solver expressions
    assert solution.value(fixed * first) == pytest.approx(8.)


def test_value_needs_no_propagation(monkeypatch):
    solution, first, second = two_parameter_solution()
    monkeypatch.setattr(solution.simulation, "run", lambda *args: pytest.fail("No propagation needed"))
    assert solution.value(first + second) == pytest.approx(-1.)


@pytest.mark.parametrize("make_quantity, error, message", [
    (lambda first, second: Parameter(1.0).variable() + first, KeyError, "outside this solution's model"),
    (lambda first, second: Symbol("unbound") + first, KeyError, "no binding"),
    (lambda first, second: Symbol(ResultSymbolKey(ResultSymbolKind.FINAL, name="x")), KeyError, r"run\(\)"),
    (lambda first, second: InputNode(None) + first, UnresolvedInputError, "empty InputNode"),
], ids=["outside-variable", "unbound", "result-symbol", "empty-handle"])
def test_value_rejects_what_the_solution_does_not_contain(make_quantity, error, message):
    solution, first, second = two_parameter_solution()
    with pytest.raises(error, match=message):
        solution.value(make_quantity(first, second))


def test_value_rejects_an_unbound_symbol_even_when_other_symbols_are_bound():
    solution, _, _ = two_parameter_solution(offset=lambda a, b: 2 * a)
    assert solution.value(Symbol("offset")) == pytest.approx(4.)
    with pytest.raises(KeyError, match="no binding"):
        solution.value(Symbol("other") + Symbol("offset"))


@pytest.mark.parametrize("quantity", ["x", None, [1.0], Parameter(1.0) >= 0], ids=["str", "none", "list", "comparison"])
def test_value_rejects_non_graph_quantities(quantity):
    solution, _, _ = two_parameter_solution()
    with pytest.raises(TypeError, match="Parameter or graph expression"):
        solution.value(quantity)


def test_str_summarizes_status_variables_objectives_and_constraints():
    from diffractix import Ray, Solver, System, ThinLens
    gap = Space(d=0.5)
    gap.d.variable(bounds=(0, 1))
    lens = ThinLens(f=0.1)
    simulation = System(gap, lens, optical_input=Ray(x=1e-3, theta=1e-3, n=1)).build()
    sim = simulation.symbolic
    solver = Solver(simulation)
    solver.target(value=sim[lens].after.x, target=0.0, label="focus")
    solver.require(sim[lens].z_end <= 0.3, label="length")
    solution = solver.solve()
    text = str(solution)
    assert text.startswith("Solution: converged, feasible")
    for expected in ("Variables", "d", "0.5", "0.3", "Objectives", "focus", "Constraints", "length", "ok"):
        assert expected in text
    assert "VIOLATED" not in text


def test_reporting_rejects_a_deferred_callable_that_returns_a_graph_node():
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))

    def residual(result):
        return p if result.final.x > 1 else result.final.x   # a Node only at the solved point

    solver.target(residual)
    solution, _ = make_solution(solver=solver)
    with pytest.raises(TypeError, match="must return numerical scalars or arrays"):
        solution.objectives


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_reporting_rejects_a_finite_residual_whose_cost_overflows():
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.target(lambda result: 1e200 if result.final.x > 1 else result.final.x, label="huge")
    solution, _ = make_solution(solver=solver)
    with pytest.raises(ValueError, match="'huge' cost is non-finite"):
        solution.objectives
