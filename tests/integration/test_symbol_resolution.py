"""Compile bindings and runtime optical quantities share canonical theta."""
import pytest
from diffractix import Constraint, Parameter, Ray, Solver, Space, System
from diffractix.graph import Symbol


@pytest.mark.parametrize("alias", [False, True])
def test_bound_expression_composes_with_runtime_state_and_solution_reporting(alias):
    distance = Parameter(0.03).variable(bounds=(0.01, 0.2))
    space = Space(d=distance)
    system = System(space, optical_input=Ray(x=0, theta=0.01, n=1))
    system.bind("distance_twice", 2 * distance)
    if alias:
        system.bind("objective_distance", Symbol("distance_twice"))
    solver = Solver(system)
    key = "objective_distance" if alias else "distance_twice"
    solver.target(lambda sim: Symbol(key) / 0.2 + sim.final.x / 0.001 - 2)
    solution = solver.solve(options={"ftol": 1e-12})
    # At d=0.1 each summand is one.
    assert solution.success and solution.feasible
    assert solution.value(distance) == pytest.approx(0.1, abs=1e-7)
    assert abs(solution.objectives[0].value) < 1e-7


@pytest.mark.parametrize("consumer", ["objective", "constraint"])
def test_unknown_symbols_and_external_variables_fail_before_native_optimization(consumer):
    position = Parameter(0.5).variable(bounds=(0, 2))
    for expression, message in ((Symbol("missing"), "unresolved Symbol"),
                                (Parameter(1).variable(), "Missing required Parameter")):
        solver = Solver(System(optical_input=Ray(x=position, theta=0, n=1)))
        if consumer == "objective":
            solver.target(expression)
        else:
            solver.target(position - 1)
            solver.require(Constraint(expression, lower_bound=0))
        with pytest.raises((ValueError, KeyError), match=message):
            solver.solve()
