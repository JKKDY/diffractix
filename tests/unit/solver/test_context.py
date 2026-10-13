"""Solver context lookups and typed Symbol keys."""

from types import SimpleNamespace
from diffractix.graph import Symbol
from diffractix.solver.solver import Solver, SolverCompileContext, SolverSymbolKey, SolverSymbolKind
from tests.helpers.solver import context, make_solver


def test_solver_compile_context_creates_typed_symbol_keys():
    element = object()
    context = SolverCompileContext()

    at = context.at(element).w
    after = context.after(element, occurrence=2).q
    z_at = context.z_at(element, occurrence=3)
    z_after = context.z_after(element, occurrence=4)
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
    assert z_at.key == SolverSymbolKey(
        kind=SolverSymbolKind.Z_AT,
        element_id=id(element),
        occurrence=3,
    )
    assert z_after.key == SolverSymbolKey(
        kind=SolverSymbolKind.Z_AFTER,
        element_id=id(element),
        occurrence=4,
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


def test_solver_symbol_keys_compare_by_all_lookup_fields():
    key = SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w")

    assert key == SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w")
    assert hash(key) == hash(SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w"))
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 124, None, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 123, 1, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "R")
    assert key != SolverSymbolKey(SolverSymbolKind.AFTER, 123, None, None, "w")
    assert key != SolverSymbolKey(SolverSymbolKind.Z, index=0)
    assert SolverSymbolKey(SolverSymbolKind.STATE, index=0, name="w") != SolverSymbolKey(
        SolverSymbolKind.STATE, index=1, name="w",
    )
    bindings = {key: 1.5}
    assert bindings[SolverSymbolKey(SolverSymbolKind.AT, 123, None, None, "w")] == 1.5


def test_solver_resolves_typed_symbol_keys():
    element = object()
    compile_context = SolverCompileContext()
    at = compile_context.at(element, occurrence=1).w
    after = compile_context.after(element, occurrence=2).q
    z_at = compile_context.z_at(element, occurrence=3)
    z_after = compile_context.z_after(element, occurrence=4)
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
        z_at=lambda target, occurrence: (target is element, occurrence, "at"),
        z_after=lambda target, occurrence: (target is element, occurrence, "after"),
    )
    solver = Solver.__new__(Solver)

    assert solver._resolve_symbol(at.key, compile_context, runtime_context) == (True, 1)
    assert solver._resolve_symbol(after.key, compile_context, runtime_context) == (True, 2)
    assert solver._resolve_symbol(z_at.key, compile_context, runtime_context) == (True, 3, "at")
    assert solver._resolve_symbol(z_after.key, compile_context, runtime_context) == (True, 4, "after")
    assert solver._resolve_symbol(z.key, compile_context, runtime_context) == 20.0
    assert solver._resolve_symbol(state.key, compile_context, runtime_context) == 30.0


def test_solver_context_exposes_runtime_z_positions():
    solver, lens, _ = make_solver()
    runtime_context = context(solver, [3.0])

    assert runtime_context.z_at(lens) == 23.0
    assert runtime_context.z_after(lens, occurrence=2) == 35.0
