from __future__ import annotations

from types import SimpleNamespace

import autograd.numpy as np
import numpy as numpy
import pytest

from diffractix.graph import Parameter
from diffractix.simulation import Simulation
from diffractix.solver import (
    Backend,
    Constraint,
    DEFAULT_FEASIBILITY_TOLERANCE,
    Objective,
    OptimizationResult,
    Solution,
    Solver,
)
from diffractix.solver.problem import Problem
from diffractix.system.system import ParameterInfo


def parameter_info(parameter, index, *, value=None, lower=-np.inf, upper=np.inf):
    return ParameterInfo(
        parameter_id=id(parameter),
        name=parameter.name,
        value=parameter.value if value is None else value,
        parameter_index=index,
        lower_bound=lower,
        upper_bound=upper,
    )


def make_simulation(parameters, *, run=None):
    simulation = object.__new__(Simulation)
    simulation.parameter_info = {
        id(parameter): parameter_info(
            parameter,
            index,
            lower=parameter.lower_bound,
            upper=parameter.upper_bound,
        )
        for index, parameter in enumerate(parameters)
    }
    simulation.compile_context = {}
    simulation.execution_context = {}
    simulation.graph = SimpleNamespace(
        initial_values=np.array([parameter.value for parameter in parameters])
    )
    simulation.run = run or (lambda theta: SimpleNamespace(theta=theta))
    return simulation


def make_problem(n_variables=1):
    return Problem(
        x0=np.zeros(n_variables),
        x_lower=np.full(n_variables, -np.inf),
        x_upper=np.full(n_variables, np.inf),
        objective=lambda theta: np.sum(theta**2),
        gradient=lambda theta: 2.0 * theta,
        objective_hessian=lambda theta: 2.0 * np.eye(n_variables),
        constraints=lambda theta: np.array([]),
        jacobian=lambda theta: np.empty((0, n_variables)),
        constraint_hessian=lambda theta, multipliers: np.zeros(
            (n_variables, n_variables)
        ),
        constraint_lower=np.array([]),
        constraint_upper=np.array([]),
    )


def make_solution(
    *,
    x=(5.0,),
    success=True,
    cost=1.25,
    message="done",
    simulation=None,
    objectives=(),
    constraints=(),
    feasibility_tolerance=DEFAULT_FEASIBILITY_TOLERANCE,
):
    if simulation is None:
        parameter = Parameter(2.0, name="p").variable()
        simulation = make_simulation((parameter,))

    return Solution(
        OptimizationResult(
            x=np.array(x),
            success=success,
            cost=cost,
            message=message,
        ),
        make_problem(len(x)),
        simulation,
        objectives,
        constraints,
        feasibility_tolerance,
    )


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


def test_solution_forwards_backend_result():
    solution = make_solution(x=(3.0,), success=False, cost=7.5, message="stopped")

    numpy.testing.assert_array_equal(solution.x, [3.0])
    assert solution.success is False
    assert solution.cost == 7.5
    assert solution.message == "stopped"


def test_solution_feasibility_tolerance_validation_and_default():
    parameter = Parameter(2.0).variable()
    simulation = make_simulation((parameter,))
    result = OptimizationResult(np.array([2.0]), True, 0.0, "done")

    with pytest.raises(ValueError, match="non-negative"):
        Solution(result, make_problem(), simulation, (), (), -1.0)

    default_solution = Solution(result, make_problem(), simulation, (), ())
    assert default_solution.feasibility_tolerance == DEFAULT_FEASIBILITY_TOLERANCE
    assert make_solution(feasibility_tolerance=1e-3).feasibility_tolerance == 1e-3


@pytest.mark.parametrize("tolerance", (np.nan, np.inf, -np.inf))
def test_solution_rejects_nonfinite_feasibility_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite|NaN|non-negative"):
        make_solution(feasibility_tolerance=tolerance)


def test_solution_rejects_backend_result_with_wrong_x_length():
    simulation = make_simulation((Parameter(2.0).variable(),))
    result = OptimizationResult(np.array([2.0, 3.0]), True, 0.0, "done")

    with pytest.raises(ValueError, match="x length.*initial_values"):
        Solution(result, make_problem(), simulation, (), ())


def test_solution_parameter_info_and_parameters_use_canonical_order():
    first = Parameter(2.0, name="first").variable()
    second = Parameter(4.0, name="second").variable()
    simulation = make_simulation((first, second))
    first_info = parameter_info(first, 1)
    second_info = parameter_info(second, 0)
    simulation.parameter_info = {
        id(first): first_info,
        id(second): second_info,
    }
    solution = make_solution(x=(5.0, 7.0), simulation=simulation)

    assert solution.parameter_info == (second_info, first_info)
    assert solution.parameter_info is solution.parameter_info
    assert solution.parameters is solution.parameters
    assert tuple(parameter.info for parameter in solution.parameters) == (
        second_info,
        first_info,
    )
    assert tuple(parameter.value for parameter in solution.parameters) == (5.0, 7.0)
    assert tuple(parameter.initial for parameter in solution.parameters) == (4.0, 2.0)
    assert tuple(parameter.delta for parameter in solution.parameters) == (1.0, 5.0)


def test_solution_value_of_and_getitem_use_solved_snapshot():
    parameter = Parameter(2.0, name="p").variable()
    simulation = make_simulation((parameter,))
    solution = make_solution(x=(5.0,), simulation=simulation)
    expression = 3.0 * parameter + 2.0
    parameter.value = 100.0

    assert solution.value_of(parameter) == pytest.approx(5.0)
    assert solution[parameter] == pytest.approx(5.0)
    assert solution[expression] == pytest.approx(17.0)


def test_solution_run_uses_solved_theta():
    parameter = Parameter(2.0).variable()
    calls = []

    def run(theta):
        calls.append(theta)
        return "simulation result"

    solution = make_solution(
        x=(5.0,),
        simulation=make_simulation((parameter,), run=run),
    )

    assert solution.run() == "simulation result"
    numpy.testing.assert_array_equal(calls[0], solution.x)


def test_solution_target_results_preserve_order_shape_weight_and_cost():
    objectives = (
        Objective(lambda context: 2.0, label="scalar"),
        Objective(lambda context: np.array([1.0, -2.0]), weight=3.0, label="vector"),
    )
    solution = make_solution(objectives=objectives)

    scalar, vector = solution.targets
    assert scalar.objective is objectives[0]
    assert scalar.value == 2.0
    assert scalar.weight == 1.0
    assert scalar.cost == 4.0
    assert vector.objective is objectives[1]
    numpy.testing.assert_array_equal(vector.value, [1.0, -2.0])
    assert vector.weight == 3.0
    assert vector.cost == 15.0
    assert solution.targets is solution.targets


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_solution_preserves_non_finite_target_values_and_costs(value):
    solution = make_solution(
        objectives=(Objective(lambda context: value),),
    )

    target = solution.targets[0]
    if np.isnan(value):
        assert np.isnan(target.value)
        assert np.isnan(target.cost)
    else:
        assert target.value == value
        assert np.isinf(target.cost)


def test_solution_empty_targets():
    solution = make_solution(objectives=())
    assert solution.targets == ()


@pytest.mark.parametrize(
    ("constraint", "value", "margin", "violation", "satisfied"),
    [
        (Constraint(lambda context: 3.0, lower_bound=2.0), 3.0, 1.0, 0.0, True),
        (Constraint(lambda context: 1.5, lower_bound=2.0), 1.5, -0.5, 0.5, False),
        (Constraint(lambda context: 3.0, upper_bound=4.0), 3.0, 1.0, 0.0, True),
        (Constraint(lambda context: 4.25, upper_bound=4.0), 4.25, -0.25, 0.25, False),
        (Constraint(lambda context: 3.0, lower_bound=1.0, upper_bound=5.0), 3.0, 2.0, 0.0, True),
        (Constraint(lambda context: 5.25, lower_bound=1.0, upper_bound=5.0), 5.25, -0.25, 0.25, False),
        (Constraint(lambda context: 3.0, lower_bound=3.0, upper_bound=3.0), 3.0, 0.0, 0.0, True),
        (Constraint(lambda context: 3.01, lower_bound=3.0, upper_bound=3.0), 3.01, -0.01, 0.01, False),
    ],
)
def test_solution_constraint_result_semantics(
    constraint,
    value,
    margin,
    violation,
    satisfied,
):
    result = make_solution(constraints=(constraint,)).constraints[0]

    assert result.constraint is constraint
    assert result.value == pytest.approx(value)
    assert result.margin == pytest.approx(margin)
    assert result.violation == pytest.approx(violation)
    assert result.satisfied is satisfied


@pytest.mark.parametrize(
    ("value", "satisfied"),
    [(3.0005, True), (3.002, False)],
)
def test_solution_feasibility_tolerance_boundary(value, satisfied):
    constraint = Constraint(
        lambda context: value,
        lower_bound=3.0,
        upper_bound=3.0,
    )
    result = make_solution(
        constraints=(constraint,),
        feasibility_tolerance=1e-3,
    ).constraints[0]

    assert result.satisfied is satisfied


def test_solution_vector_constraint_preserves_shape_and_requires_all_values():
    constraint = Constraint(
        lambda context: np.array([2.0, 4.25]),
        lower_bound=1.0,
        upper_bound=4.0,
    )
    result = make_solution(constraints=(constraint,)).constraints[0]

    assert result.value.shape == (2,)
    numpy.testing.assert_allclose(result.margin, [1.0, -0.25])
    numpy.testing.assert_allclose(result.violation, [0.0, 0.25])
    assert result.satisfied is False


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_constraint_values_are_never_reported_feasible(value):
    constraint = Constraint(
        lambda context: value,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    solution = make_solution(constraints=(constraint,))

    result = solution.constraints[0]
    assert result.satisfied is False
    assert solution.violations == (result,)
    assert solution.feasible is False


@pytest.mark.parametrize("x", [(), (3.0, 4.0)])
def test_solution_rejects_backend_theta_length_mismatch(x):
    parameter = Parameter(2.0, name="p").variable()
    simulation = make_simulation((parameter,))

    with pytest.raises(ValueError, match="length|parameter|theta"):
        make_solution(x=x, simulation=simulation)


def test_solution_empty_constraints_are_feasible():
    solution = make_solution(constraints=())

    assert solution.constraints == ()
    assert solution.violations == ()
    assert solution.feasible is True


def test_solution_violations_preserve_unsatisfied_constraint_order():
    constraints = (
        Constraint(lambda context: 2.0, lower_bound=1.0),
        Constraint(lambda context: 0.0, lower_bound=1.0, label="first violation"),
        Constraint(lambda context: 3.0, upper_bound=2.0, label="second violation"),
    )
    solution = make_solution(constraints=constraints)

    assert tuple(result.constraint for result in solution.violations) == constraints[1:]
    assert solution.violations is solution.violations
    assert solution.constraints is solution.constraints
    assert solution.feasible is False


@pytest.mark.parametrize(
    ("success", "constraint", "feasible"),
    [
        (True, Constraint(lambda context: 0.0, lower_bound=1.0), False),
        (False, Constraint(lambda context: 2.0, lower_bound=1.0), True),
    ],
)
def test_solution_success_and_feasibility_are_independent(
    success,
    constraint,
    feasible,
):
    solution = make_solution(success=success, constraints=(constraint,))

    assert solution.success is success
    assert solution.feasible is feasible
    assert solution.feasible is solution.feasible


def test_solution_shares_one_runtime_simulation_result():
    parameter = Parameter(2.0).variable()
    element = object()
    calls = 0

    class Result:
        def at(self, location, occurrence=None):
            assert location is element
            return SimpleNamespace(w=3.0)

    def run(theta):
        nonlocal calls
        calls += 1
        return Result()

    simulation = make_simulation((parameter,), run=run)
    objective = Objective(lambda context: context.at(element).w)
    constraint = Constraint(lambda context: context.at(element).w, lower_bound=2.0)
    solution = make_solution(
        simulation=simulation,
        objectives=(objective,),
        constraints=(constraint,),
    )

    solution.targets
    solution.constraints
    solution.violations
    solution.feasible
    solution.run()

    assert calls == 1


def test_solution_pure_graph_access_does_not_run_simulation():
    parameter = Parameter(2.0).variable()
    calls = 0

    def run(theta):
        nonlocal calls
        calls += 1

    solution = make_solution(
        x=(5.0,),
        simulation=make_simulation((parameter,), run=run),
    )

    assert solution[parameter] == 5.0
    assert solution[2.0 * parameter + 1.0] == 11.0
    assert calls == 0


def test_solution_does_not_mutate_declarative_parameters():
    parameter = Parameter(2.0, name="p").variable()
    simulation = make_simulation((parameter,))
    solution = make_solution(x=(5.0,), simulation=simulation)

    solution.parameters
    solution[parameter]
    solution.targets
    solution.constraints
    solution.run()

    assert parameter.value == 2.0
    assert parameter.is_variable is True
    assert solution[parameter] == 5.0
    assert solution.parameters[0].value == 5.0
