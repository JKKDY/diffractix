from dataclasses import dataclass
from types import SimpleNamespace

import autograd.numpy as np
import numpy as numpy
import pytest
from autograd import grad

from diffractix.graph import Parameter, Symbol
from diffractix.simulation import Simulation
from diffractix.solver.solver import (
    Solver,
    SolverCompileContext,
    SolverContext,
    SolverSymbolKey,
    SolverSymbolKind,
)
from diffractix.solver.objective import Objective
from diffractix.solver.constraint import Constraint


@dataclass
class FakeState:
    w: object
    R: object
    q: object


class FakeResult:
    def __init__(self, theta, element):
        self.theta = theta
        self.element = element
        self.z = np.stack([0.0 * theta[0], theta[0] + 10.0])
        self.states = (
            FakeState(theta[0] + 1.0, theta[0] + 2.0, theta[0] + 3.0),
            FakeState(theta[0] + 2.0, theta[0] + 3.0, theta[0] + 4.0),
        )

    def at(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return FakeState(
            self.theta[0] + 1.0 + occurrence,
            self.theta[0] + 2.0 + occurrence,
            self.theta[0] + 3.0 + occurrence,
        )

    def after(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return FakeState(
            self.theta[0] + 2.0 + occurrence,
            self.theta[0] + 3.0 + occurrence,
            self.theta[0] + 4.0 + occurrence,
        )


class FakeElement:
    pass


def make_solver(value=2.0):
    element = FakeElement()
    parameter = Parameter(value, name="p").variable()
    element.p = parameter

    simulation = object.__new__(Simulation)
    simulation.parameter_info = {
        id(parameter): SimpleNamespace(
            value=value,
            parameter_index=0,
        ),
    }
    simulation.simulation_context = {}
    simulation.run = lambda theta: FakeResult(theta, element)

    return Solver(simulation), element, parameter


def context(solver, theta):
    return SolverContext(np.array(theta), solver.simulation.run)


def make_solver_with_reordered_parameters():
    solver, element, first = make_solver()
    second = Parameter(4.0, name="second").variable()

    solver.simulation.parameter_info = {
        id(first): SimpleNamespace(value=2.0, parameter_index=1),
        id(second): SimpleNamespace(value=4.0, parameter_index=0),
    }

    return solver, element, first, second


def test_solver_compile_context_creates_typed_symbol_keys():
    element = object()
    context = SolverCompileContext()

    at = context.at(element).w
    after = context.after(element, occurrence=2).q
    z = context.z[3]
    state = context.states[4].w

    assert isinstance(at, Symbol)
    assert at.key == SolverSymbolKey(
        kind=SolverSymbolKind.AT,
        element_id=id(element),
        name="w",
    )
    assert after.key == SolverSymbolKey(
        kind=SolverSymbolKind.AFTER,
        element_id=id(element),
        occurrence=2,
        name="q",
    )
    assert z.key == SolverSymbolKey(
        kind=SolverSymbolKind.Z,
        index=3,
    )
    assert state.key == SolverSymbolKey(
        kind=SolverSymbolKind.STATE,
        index=4,
        name="w",
    )
    assert context._target_elements[id(element)] is element


def test_solver_symbol_keys_are_hashable_runtime_binding_keys():
    key = SolverSymbolKey(
        kind=SolverSymbolKind.STATE,
        index=2,
        name="w",
    )
    bindings = {key: 1.5}

    assert bindings[key] == 1.5


def test_solver_symbol_keys_compare_by_all_lookup_fields():
    key = SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w")

    assert key == SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w")
    assert hash(key) == hash(SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w"))
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 124, None, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 123, 1, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "R")
    assert key != SolverSymbolKey(SolverSymbolKind.AFTER, 123, None, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.Z, index=0)


def test_solver_resolves_typed_symbol_keys():
    element = object()
    compile_context = SolverCompileContext()
    at = compile_context.at(element, occurrence=1).w
    after = compile_context.after(element, occurrence=2).q
    z = compile_context.z[1]
    state = compile_context.states[0].w

    runtime_context = SimpleNamespace(
        z=(10.0, 20.0),
        states=(SimpleNamespace(w=30.0),),
        at=lambda target, occurrence: SimpleNamespace(
            w=(target is element, occurrence),
        ),
        after=lambda target, occurrence: SimpleNamespace(
            q=(target is element, occurrence),
        ),
    )
    solver = Solver.__new__(Solver)

    assert solver._resolve_symbol(at.key, compile_context, runtime_context) == (True, 1)
    assert solver._resolve_symbol(after.key, compile_context, runtime_context) == (True, 2)
    assert solver._resolve_symbol(z.key, compile_context, runtime_context) == 20.0
    assert solver._resolve_symbol(state.key, compile_context, runtime_context) == 30.0


def test_compile_direct_node_objective():
    solver, _, parameter = make_solver()
    solver.target(parameter - 3.0)

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [5.0])) == 2.0


def test_direct_node_uses_canonical_theta_when_local_order_differs():
    solver, _, first, second = make_solver_with_reordered_parameters()
    solver.target(first + 10.0 * second)

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [3.0, 5.0])) == 35.0


def test_compile_traced_objective_with_runtime_symbols():
    solver, lens, parameter = make_solver()
    solver.objectives = [
        Objective(
            lambda ctx:
                parameter
                + ctx.at(lens).w
                + ctx.after(lens).w
                + ctx.z[1]
                + ctx.states[1].w
        ),
    ]

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [3.0])) == 30.0


def test_compile_traced_objective_mixes_all_runtime_sources():
    solver, lens, parameter = make_solver()
    solver.target(
        lambda ctx:
            parameter
            + ctx.at(lens).w
            + ctx.after(lens).R
            + ctx.z[1]
            + ctx.states[1].q,
    )

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [3.0])) == 33.0


def test_at_symbols_preserve_property_identity_and_repeated_uses():
    solver, lens, _ = make_solver()
    solver.target(
        lambda ctx: ctx.at(lens).w + ctx.at(lens).R,
        lambda ctx: ctx.at(lens).w + ctx.at(lens).w,
    )

    first, second = solver._compile_objectives()
    runtime_context = context(solver, [3.0])

    assert first.evaluate(runtime_context) == 9.0
    assert second.evaluate(runtime_context) == 8.0


def test_at_symbols_distinguish_occurrences():
    solver, lens, _ = make_solver()
    solver.target(
        lambda ctx:
            ctx.at(lens, occurrence=0).w
            - ctx.at(lens, occurrence=1).w,
    )

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [3.0])) == -1.0


def test_after_z_and_states_resolve_independently():
    solver, lens, parameter = make_solver()
    solver.target(
        lambda ctx: ctx.after(lens).q + ctx.z[0] + parameter,
        lambda ctx: ctx.states[0].w + ctx.states[1].R,
    )

    first, second = solver._compile_objectives()
    runtime_context = context(solver, [3.0])

    assert first.evaluate(runtime_context) == 10.0
    assert second.evaluate(runtime_context) == 10.0


def test_z_indices_produce_distinct_runtime_values():
    solver, _, parameter = make_solver()
    solver.target(lambda ctx: parameter + ctx.z[0], lambda ctx: parameter + ctx.z[1])

    first, second = solver._compile_objectives()
    runtime_context = context(solver, [3.0])

    assert first.evaluate(runtime_context) == 3.0
    assert second.evaluate(runtime_context) == 16.0


def test_compile_traced_objective_preserves_occurrence():
    solver, lens, _ = make_solver()
    solver.objectives = [
        Objective(lambda ctx: ctx.at(lens, occurrence=2).w),
    ]

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [3.0])) == 6.0


def test_compile_constant_callable():
    solver, _, _ = make_solver()
    solver.objectives = [Objective(lambda ctx: 7.0)]

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [100.0])) == 7.0


def test_constant_callable_does_not_run_simulation_during_compilation():
    solver, lens, _ = make_solver()
    calls = 0

    def run(theta):
        nonlocal calls
        calls += 1
        return FakeResult(theta, lens)

    solver.simulation.run = run
    solver.target(lambda ctx: 7.0)

    solver._compile_objectives()

    assert calls == 0


def test_dynamic_python_branch_falls_back_to_interpreter():
    solver, lens, parameter = make_solver()

    def objective(ctx):
        if ctx.at(lens).w > 3.0:
            return (parameter - 5.0) ** 2
        return parameter + 10.0

    solver.objectives = [Objective(objective)]

    compiled, = solver._compile_objectives()

    assert compiled.evaluate(context(solver, [1.0])) == 11.0
    assert compiled.evaluate(context(solver, [4.0])) == 1.0


def test_dynamic_callable_can_have_multiple_python_branches():
    solver, lens, parameter = make_solver()

    def objective(ctx):
        width = ctx.at(lens).w

        if width > 5.0:
            return abs(parameter - 10.0)

        if width > 2.0:
            return parameter**2

        return width

    solver.objectives = [Objective(objective)]

    compiled, = solver._compile_objectives()

    assert compiled.evaluate(context(solver, [0.0])) == 1.0
    assert compiled.evaluate(context(solver, [2.0])) == 4.0
    assert compiled.evaluate(context(solver, [6.0])) == 4.0


def test_dynamic_callable_supports_node_and_numerical_results():
    solver, lens, parameter = make_solver()

    def objective(ctx):
        if ctx.at(lens).w > 3.0:
            return parameter * 2.0
        return 5.0

    solver.target(objective)

    compiled, = solver._compile_objectives()

    assert compiled.evaluate(context(solver, [1.0])) == 5.0
    assert compiled.evaluate(context(solver, [4.0])) == 8.0


def test_dynamic_callable_falls_back_for_raw_symbol_truth_testing():
    solver, lens, parameter = make_solver()

    def objective(ctx):
        if ctx.at(lens).w:
            return parameter * 2.0
        return 5.0

    solver.target(objective)

    compiled, = solver._compile_objectives()

    assert compiled.evaluate(context(solver, [-1.0])) == 5.0
    assert compiled.evaluate(context(solver, [2.0])) == 4.0


def test_compile_direct_node_constraint():
    solver, _, parameter = make_solver()
    solver.require(
        Constraint(
            parameter,
            lower_bound=1.0,
            upper_bound=5.0,
        ),
    )

    constraint, = solver._compile_constraints()

    assert constraint.lower_bound == 1.0
    assert constraint.upper_bound == 5.0
    assert constraint.evaluate(context(solver, [3.0])) == 3.0


def test_compile_traced_runtime_constraint():
    solver, lens, parameter = make_solver()
    solver.constraints = [
        Constraint(
            lambda ctx: parameter - 2.0 * ctx.at(lens).w,
            lower_bound=0.0,
        ),
    ]

    constraint, = solver._compile_constraints()

    assert constraint.evaluate(context(solver, [3.0])) == -5.0
    assert constraint.lower_bound == 0.0


def test_compile_multiple_constraints_preserves_order_and_bounds():
    solver, lens, parameter = make_solver()
    solver.require(
        Constraint(parameter, lower_bound=1.0, upper_bound=5.0, label="direct"),
        Constraint(
            lambda ctx: parameter - 2.0 * ctx.at(lens).w,
            lower_bound=0.0,
            label="traced",
        ),
    )

    direct, traced = solver._compile_constraints()
    runtime_context = context(solver, [3.0])

    assert (direct.lower_bound, direct.upper_bound, direct.label) == (1.0, 5.0, "direct")
    assert (traced.lower_bound, traced.upper_bound, traced.label) == (0.0, np.inf, "traced")
    assert direct.evaluate(runtime_context) == 3.0
    assert traced.evaluate(runtime_context) == -5.0


def test_compile_dynamic_constraint():
    solver, lens, parameter = make_solver()

    def constraint(ctx):
        if ctx.after(lens).w > 5.0:
            return parameter * 2.0
        return parameter - 1.0

    solver.constraints = [
        Constraint(
            constraint,
            lower_bound=-2.0,
            upper_bound=8.0,
        ),
    ]

    compiled, = solver._compile_constraints()

    assert compiled.evaluate(context(solver, [1.0])) == 0.0
    assert compiled.evaluate(context(solver, [4.0])) == 8.0


def test_snapshot_fixed_value_overrides_live_parameter_mutation():
    solver, _, parameter = make_solver()
    solver.simulation.parameter_info[id(parameter)] = SimpleNamespace(
        value=2.0,
        parameter_index=None,
    )
    parameter.value = 100.0
    solver.target(parameter * 3.0)

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [999.0])) == 6.0


def test_snapshot_variable_membership_overrides_live_fixed_state():
    solver, _, parameter = make_solver()
    parameter.fixed()
    solver.target(parameter * 2.0)

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [4.0])) == 8.0


def test_snapshot_fixed_membership_overrides_live_variable_state():
    solver, _, parameter = make_solver()
    solver.simulation.parameter_info[id(parameter)] = SimpleNamespace(
        value=2.0,
        parameter_index=None,
    )
    parameter.variable()
    solver.target(parameter * 2.0)

    objective, = solver._compile_objectives()

    assert objective.evaluate(context(solver, [4.0])) == 4.0


def test_tracing_does_not_run_simulation():
    solver, lens, _ = make_solver()
    calls = 0

    def run(theta):
        nonlocal calls
        calls += 1
        return FakeResult(theta, lens)

    solver.simulation.run = run
    solver.objectives = [Objective(lambda ctx: ctx.at(lens).w)]
    solver.constraints = [
        Constraint(lambda ctx: ctx.after(lens).R, lower_bound=0.0),
    ]

    solver._compile_objectives()
    solver._compile_constraints()

    assert calls == 0


def test_runtime_context_runs_simulation_only_once():
    solver, lens, _ = make_solver()
    calls = 0

    def run(theta):
        nonlocal calls
        calls += 1
        return FakeResult(theta, lens)

    solver.simulation.run = run
    solver.objectives = [
        Objective(lambda ctx: ctx.at(lens).w),
        Objective(lambda ctx: ctx.after(lens).R),
        Objective(lambda ctx: ctx.z[1]),
        Objective(lambda ctx: ctx.states[1].q),
    ]

    objectives = solver._compile_objectives()
    runtime_context = context(solver, [3.0])
    values = [objective.evaluate(runtime_context) for objective in objectives]

    assert values == [4.0, 6.0, 13.0, 7.0]
    assert calls == 1


def test_different_runtime_contexts_do_not_share_simulation_results():
    solver, lens, _ = make_solver()
    calls = 0

    def run(theta):
        nonlocal calls
        calls += 1
        return FakeResult(theta, lens)

    solver.simulation.run = run
    solver.target(lambda ctx: ctx.at(lens).w)
    objective, = solver._compile_objectives()

    first_context = context(solver, [1.0])
    second_context = context(solver, [4.0])

    assert objective.evaluate(first_context) == 2.0
    assert objective.evaluate(first_context) == 2.0
    assert objective.evaluate(second_context) == 5.0
    assert objective.evaluate(second_context) == 5.0
    assert calls == 2


def test_traced_hybrid_expression_is_autograd_differentiable():
    solver, lens, parameter = make_solver()
    solver.target(lambda ctx: parameter**2 + 3.0 * ctx.at(lens).w)
    objective, = solver._compile_objectives()

    derivative = grad(
        lambda theta: objective.evaluate(
            SolverContext(theta, solver.simulation.run),
        )
    )

    numpy.testing.assert_allclose(derivative(np.array([4.0])), [11.0])


def test_dynamic_branch_is_autograd_differentiable_away_from_boundary():
    solver, lens, parameter = make_solver()

    def objective(ctx):
        if ctx.at(lens).w > 0.0:
            return parameter**2
        return parameter + 10.0

    solver.target(objective)
    compiled, = solver._compile_objectives()

    derivative = grad(
        lambda theta: compiled.evaluate(
            SolverContext(theta, solver.simulation.run),
        )
    )

    numpy.testing.assert_allclose(derivative(np.array([2.0])), [4.0])


def test_compile_callable_does_not_swallow_unrelated_errors():
    solver, _, _ = make_solver()

    def broken(ctx):
        return 1 + "not a number"

    solver.objectives = [Objective(broken)]

    with pytest.raises(TypeError):
        solver._compile_objectives()


def test_unknown_element_error_propagates_at_runtime():
    solver, _, _ = make_solver()
    missing = FakeElement()
    solver.target(lambda ctx: ctx.at(missing).w)
    objective, = solver._compile_objectives()

    with pytest.raises(KeyError, match="Unknown element"):
        objective.evaluate(context(solver, [2.0]))


def test_unknown_runtime_property_error_propagates_at_runtime():
    solver, lens, _ = make_solver()
    solver.target(lambda ctx: ctx.at(lens).does_not_exist)
    objective, = solver._compile_objectives()

    with pytest.raises(AttributeError, match="does_not_exist"):
        objective.evaluate(context(solver, [2.0]))


def test_direct_node_cannot_contain_solver_runtime_symbol():
    solver, _, _ = make_solver()
    symbol = Symbol(
        SolverSymbolKey(
            kind=SolverSymbolKind.Z,
            index=0,
        )
    )
    solver.objectives = [Objective(symbol)]

    with pytest.raises(ValueError, match="runtime Symbols"):
        solver._compile_objectives()


def test_empty_objective_and_constraint_collections_compile_to_empty_tuples():
    solver, _, _ = make_solver()

    assert solver._compile_objectives() == ()
    assert solver._compile_constraints() == ()


def test_multiple_objectives_compile_and_evaluate_in_order():
    solver, lens, parameter = make_solver()

    def dynamic(ctx):
        if ctx.at(lens).w > 3.0:
            return parameter * 2.0
        return parameter + 10.0

    solver.target(
        parameter - 3.0,
        Objective(lambda ctx: ctx.at(lens).w, weight=2.0, label="at"),
        Objective(lambda ctx: ctx.z[1], label="z"),
        Objective(dynamic, label="dynamic"),
        Objective(lambda ctx: 7.0, label="constant"),
    )

    objectives = solver._compile_objectives()
    runtime_context = context(solver, [3.0])

    assert [objective.evaluate(runtime_context) for objective in objectives] == [0.0, 4.0, 13.0, 6.0, 7.0]
    assert [objective.weight for objective in objectives] == [1.0, 2.0, 1.0, 1.0, 1.0]
    assert [objective.label for objective in objectives] == [None, "at", "z", "dynamic", "constant"]


def test_multiple_constraints_compile_and_evaluate_in_order():
    solver, lens, parameter = make_solver()

    def dynamic(ctx):
        if ctx.after(lens).w > 5.0:
            return parameter * 2.0
        return parameter - 1.0

    solver.require(
        Constraint(parameter, lower_bound=1.0, upper_bound=5.0, label="direct"),
        Constraint(lambda ctx: parameter - 2.0 * ctx.at(lens).w, lower_bound=0.0, label="traced"),
        Constraint(dynamic, lower_bound=-2.0, upper_bound=8.0, label="dynamic"),
    )

    constraints = solver._compile_constraints()
    runtime_context = context(solver, [4.0])

    assert [constraint.evaluate(runtime_context) for constraint in constraints] == [4.0, -6.0, 8.0]
    assert [constraint.lower_bound for constraint in constraints] == [1.0, 0.0, -2.0]
    assert [constraint.upper_bound for constraint in constraints] == [5.0, np.inf, 8.0]
    assert [constraint.label for constraint in constraints] == ["direct", "traced", "dynamic"]


def test_public_normalization_composes_static_and_runtime_expressions():
    solver, lens, parameter = make_solver()

    solver.target(
        parameter - 3.0,
        Objective(lambda ctx: ctx.at(lens).w - 4.0, weight=2.0, label="beam"),
        Objective(lambda ctx: parameter + ctx.after(lens).R + ctx.z[1]),
    )
    solver.require(
        parameter >= 1.0,
        Constraint(
            lambda ctx: parameter - 3.0 * ctx.at(lens).w,
            lower_bound=0.0,
            label="aperture",
        ),
    )

    objectives = solver._compile_objectives()
    constraints = solver._compile_constraints()
    runtime_context = context(solver, [3.0])

    assert [objective.evaluate(runtime_context) for objective in objectives] == [0.0, 0.0, 22.0]
    assert [objective.weight for objective in objectives] == [1.0, 2.0, 1.0]
    assert [constraint.evaluate(runtime_context) for constraint in constraints] == [3.0, -9.0]
    assert [(constraint.lower_bound, constraint.upper_bound) for constraint in constraints] == [
        (1.0, np.inf),
        (0.0, np.inf),
    ]
