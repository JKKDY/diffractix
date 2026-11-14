"""Objective tracing preserves residuals, weights and fallback ordering."""

import numpy as np
import pytest

from diffractix import Parameter, Ray, Solver, System
from diffractix.backend import grad, xp
from diffractix.graph import Symbol, optimize, remap_parameters
from diffractix.simulation.result import ResultSymbolKey, ResultSymbolKind, SimulationResultTracer
from diffractix.solver.objective import Objective, prepare_objectives


def make_tracer():
    return SimulationResultTracer((), Ray(x=0, theta=0, n=1).numeric())


def test_direct_and_traced_residuals_keep_weights_bindings_and_canonical_indices():
    first, second = Parameter(1).variable(), Parameter(2).variable()
    program, descriptors, callables = prepare_objectives(
        (
            Objective(second - first, weight=2, label="direct"),
            Objective(lambda result: Symbol("offset") + first * result.final.x, weight=3, label="traced"),
        ),
        tracer=make_tracer(), bindings={"offset": 4},
    )
    program = optimize(remap_parameters(program, {id(first): 0, id(second): 1}))
    final_x = ResultSymbolKey(ResultSymbolKind.FINAL, name="x")
    assert program.evaluate([3, 5], {final_x: 2}) == (2, 10)
    assert [(item.position, item.weight, item.label) for item in descriptors] == [
        (0, 2, "direct"), (1, 3, "traced"),
    ]
    assert callables == ()
    derivative = grad(lambda theta: program.evaluate(theta, {final_x: theta[0]})[1])
    np.testing.assert_allclose(derivative(xp.array([3.0, 5.0])), [6, 0])


def test_dynamic_objectives_preserve_original_callable_and_declaration_position():
    parameter = Parameter(1).variable()

    def dynamic(result):
        return parameter - 5 if result.final.x > 0 else parameter + 10

    objective = Objective(dynamic, weight=4, label="dynamic")
    program, descriptors, callables = prepare_objectives(
        (parameter, objective, lambda result: 7.0), tracer=make_tracer(), bindings={},
    )
    assert program.evaluate([2]) == (2, 7)
    assert [item.position for item in descriptors] == [0, 2]
    assert callables == ((1, objective),)
    assert callables[0][1].quantity is dynamic


def test_array_residual_callable_is_preserved():
    objective = Objective(lambda result: xp.array([1.0, -2.0]), weight=3)
    program, descriptors, callables = prepare_objectives((objective,), tracer=make_tracer(), bindings={})
    assert program.evaluate([]) == ()
    assert descriptors == ()
    assert callables == ((0, objective),)
    np.testing.assert_array_equal(objective.quantity(None), [1, -2])


def test_objective_tracing_propagates_user_errors_and_rejects_unbound_symbols():
    with pytest.raises(TypeError):
        prepare_objectives((lambda result: 1 + "invalid",), tracer=make_tracer(), bindings={})
    with pytest.raises(ValueError, match="Objective 0.*missing"):
        prepare_objectives((Symbol("missing"),), tracer=make_tracer(), bindings={})


def test_prepared_fixed_objective_is_a_snapshot_and_empty_objectives_work():
    parameter = Parameter(2)
    program, descriptors, callables = prepare_objectives((parameter * 3,), tracer=make_tracer(), bindings={})
    parameter.value = 100
    assert program.evaluate([]) == (6,)
    assert len(descriptors) == 1
    assert callables == ()
    empty, descriptors, callables = prepare_objectives((), tracer=make_tracer(), bindings={})
    assert empty.evaluate([]) == ()
    assert descriptors == callables == ()


def test_solve_optimizes_full_constraints_and_prepares_objectives(monkeypatch):
    import diffractix.solver.solver as solver_module

    parameter = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.require(parameter >= 0, parameter <= 1)
    solver.target(parameter - 0.25)
    optimized = []

    def record_optimization(program):
        result = optimize(program)
        optimized.append(result)
        return result

    # Isolate assembly from the reducer's separately tested proof algorithms.
    monkeypatch.setattr(solver_module, "reduce_constraints", lambda program, descriptors, **bounds: ((), ()))
    monkeypatch.setattr(solver_module, "optimize", record_optimization)
    from tests.helpers.backend import capture_problem
    capture_problem(solver)
    assert len(optimized) == 2
    assert optimized[0].evaluate([0.5]) == (0.5, 0.5)
    assert optimized[1].evaluate([0.5]) == (0.25,)


def test_solver_objectives_cannot_introduce_external_variable_parameters():
    parameter, external = Parameter(0.5).variable(), Parameter(1).variable()
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    solver.target(external)
    with pytest.raises(ValueError, match="Missing required Parameter"):
        solver.solve()
