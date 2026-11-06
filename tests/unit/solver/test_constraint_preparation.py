"""Preparing direct constraints against the Simulation's canonical theta."""

import pytest

from diffractix import Constraint, Parameter, Ray, Solver, Space, System
from diffractix.graph import Symbol


def make_solver():
    first = Parameter(1.0).variable()
    second = Parameter(2.0).variable()
    system = System(Space(d=first + second), optical_input=Ray(x=0, theta=0, n=1))
    system.require(second - first >= 0)
    return Solver(system.build()), first, second


def test_preparation_fuses_system_and_solver_graphs_with_canonical_indices():
    solver, first, second = make_solver()
    solver.require(Constraint(first - 2 * second, upper_bound=1, label="solver"))
    program, descriptors, callables = solver._prepare_constraints()

    theta = [0.0] * len(program.parameter_indices)
    theta[program.parameter_indices[id(first)]] = 3.0
    theta[program.parameter_indices[id(second)]] = 5.0
    assert program.evaluate(theta) == (2.0, 8.0, -7.0)
    assert [item.label for item in descriptors] == [None, None, "solver"]
    assert descriptors[-1].upper_bound == 1
    assert callables == ()


def test_preparation_keeps_callable_constraints_deferred_in_order():
    solver, first, _ = make_solver()
    function = lambda context: context.z[0]
    solver.require(Constraint(function, lower_bound=0), first >= 0)
    program, descriptors, callables = solver._prepare_constraints()
    assert callables[0][1].evaluate is function
    assert callables[0][0] < descriptors[-1].position
    assert len(program._outputs) == len(descriptors)


def test_preparation_rejects_unregistered_parameters_and_unresolved_symbols():
    solver, _, _ = make_solver()
    solver.require(Parameter(1).variable() >= 0)
    with pytest.raises(ValueError, match="Missing required Parameter"):
        solver._prepare_constraints()
    solver.constraints.clear()
    solver.require(Symbol("missing") >= 0)
    with pytest.raises(ValueError, match="missing"):
        solver._prepare_constraints()
