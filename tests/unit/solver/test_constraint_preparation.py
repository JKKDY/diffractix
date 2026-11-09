"""Preparing direct constraints against the Simulation's canonical theta."""

import pytest

from diffractix import Constraint, Parameter, Ray, Solver, Space, System
from diffractix.graph import Symbol
from diffractix.simulation.result import ResultSymbolKey, ResultSymbolKind
from diffractix.solver.constraint import prepare_constraints



def make_solver():
    first = Parameter(1.0).variable()
    second = Parameter(2.0).variable()
    system = System(Space(d=first + second), optical_input=Ray(x=0, theta=0, n=1))
    system.require(second - first >= 0)
    return Solver(system.build()), first, second


def test_preparation_fuses_system_and_solver_graphs_with_canonical_indices():
    solver, first, second = make_solver()
    solver.require(Constraint(first - 2 * second, upper_bound=1, label="solver"))
    program, descriptors, callables = solver._finalize_constraints()

    theta = [0.0] * len(program.parameter_indices)
    theta[program.parameter_indices[id(first)]] = 3.0
    theta[program.parameter_indices[id(second)]] = 5.0
    assert program.evaluate(theta) == (2.0, 8.0, -7.0)
    assert [item.label for item in descriptors] == [None, None, "solver"]
    assert descriptors[-1].upper_bound == 1
    assert callables == ()


def test_preparation_keeps_callable_constraints_deferred_in_order():
    solver, first, _ = make_solver()
    function = lambda result: 1.0 if result.final.x > 0 else 0.0
    solver.require(Constraint(function, lower_bound=0), first >= 0)
    program, descriptors, callables = solver._finalize_constraints()
    assert callables[0][1].evaluate is function
    assert callables[0][0] < descriptors[-1].position
    assert len(program._outputs) == len(descriptors)


def test_preparation_rejects_unregistered_parameters_and_unresolved_symbols():
    # Keep the model's parameters alive: a freed Parameter's id() can be reused by
    # the new one below, which would then look like a registered parameter.
    solver, first, second = make_solver()
    outsider = Parameter(1).variable()
    solver.require(outsider >= 0)
    with pytest.raises(ValueError, match="Missing required Parameter"):
        solver._finalize_constraints()
    assert id(outsider) not in (id(first), id(second))
    solver, _, _ = make_solver()
    solver.require(Symbol("missing") >= 0)
    with pytest.raises(ValueError, match="missing"):
        solver._finalize_constraints()


def test_traces_simulation_and_solver_callables_and_keeps_dynamic_order():
    parameter = Parameter(0.1).variable()
    space = Space(d=parameter)
    system = System(space, optical_input=Ray(x=1.0, theta=2.0, n=1.0))
    system.require(Constraint(lambda result: result[space].after.x, lower_bound=0, label="system"))
    solver = Solver(system.build())
    solver.require(
        Constraint(lambda result: result[space].after.x + result.final.x, upper_bound=10, label="traced"),
        Constraint(lambda result: 1.0 if result.final.x > 0 else -1.0, label="dynamic"),
        Constraint(lambda result: result[space].z_end + parameter, label="position"),
    )
    program, descriptors, callables = solver._finalize_constraints()
    after = ResultSymbolKey(ResultSymbolKind.AFTER, 0, "x")
    final = ResultSymbolKey(ResultSymbolKind.FINAL, name="x")
    z_end = ResultSymbolKey(ResultSymbolKind.Z_END, 0)
    assert list(program.runtime_symbols) == [after, final, z_end]
    assert program.evaluate([0.2], {after: 1.4, final: 1.4, z_end: 0.2}) == pytest.approx((1.4, 0.2, 2.8, 0.4))
    assert [item.label for item in descriptors] == ["system", None, "traced", "position"]
    assert descriptors[-2].position < callables[0][0] < descriptors[-1].position
    assert callables[0][1].label == "dynamic"


def test_trace_does_not_swallow_unrelated_user_errors():
    solver, _, _ = make_solver()
    solver.require(Constraint(lambda result: 1 + "invalid"))
    with pytest.raises(TypeError):
        solver._finalize_constraints()


def test_satisfied_constant_callable_is_removed():
    solver, _, _ = make_solver()
    solver.require(Constraint(lambda result: 7.0, label="constant"))
    program, descriptors, callables = solver._finalize_constraints()
    assert program.evaluate([1.0, 2.0]) == (1.0, 3.0)
    assert all(item.label != "constant" for item in descriptors)
    assert callables == ()


def test_system_callable_is_traced_at_build_and_not_retraced_by_solver():
    parameter = Parameter(0.1).variable()
    space = Space(d=parameter)
    calls = []

    def constraint(result):
        calls.append(1)
        return result[space].after.x + parameter

    system = System(space, optical_input=Ray(x=1.0, theta=2.0, n=1.0))
    system.require(Constraint(constraint, label="system callable"))
    simulation = system.build()
    assert calls == [1]
    assert simulation.callable_constraints == ()
    parameter.value = 0.3
    solver = Solver(simulation)
    program, descriptors, _ = solver._finalize_constraints()
    solver._finalize_constraints()
    assert calls == [1]
    key = ResultSymbolKey(ResultSymbolKind.AFTER, 0, "x")
    assert program.evaluate([0.2], {key: 1.4})[0] == pytest.approx(1.6)
    assert descriptors[0].label == "system callable"


def test_system_trace_rejects_external_variable_parameters():
    extra = Parameter(1.0).variable()
    system = System(Space(d=0.1), optical_input=Ray(x=0, theta=0, n=1))
    system.require(Constraint(lambda result: result.final.x + extra))
    with pytest.raises(ValueError, match="outside the optical model"):
        system.build()


def test_preparation_is_independent_of_canonical_theta():
    first, second = Parameter(1).variable(), Parameter(2).variable()
    program, descriptors, callables = prepare_constraints(
        (second - first >= 0,), tracer=None, bindings={},
    )
    assert program.parameter_indices == {id(second): 0, id(first): 1}
    assert program.evaluate([5, 3]) == (2,)
    assert descriptors[0].position == 0
    assert callables == ()


def test_solver_constraints_use_simulation_bindings():
    parameter = Parameter(0.1).variable()
    system = System(Space(d=parameter), optical_input=Ray(x=0, theta=0, n=1))
    system.bind("distance", parameter)
    simulation = system.build()
    system.bind("distance", 99.0)

    solver = Solver(simulation)
    solver.require(Constraint(Symbol("distance") * 2, upper_bound=1, label="bound symbol"))
    program, descriptors, callables = solver._finalize_constraints()

    assert program.evaluate([0.2])[-1] == pytest.approx(0.4)
    assert program.runtime_symbols == {}
    assert descriptors[-1].label == "bound symbol"
    assert callables == ()
    assert simulation.bindings["distance"] is parameter
    with pytest.raises(TypeError):
        simulation.bindings["distance"] = 99.0
