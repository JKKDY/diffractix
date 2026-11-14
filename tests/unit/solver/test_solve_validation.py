"""Validate solve inputs before compiling the optimization model."""

import numpy as np
import pytest

from diffractix import Parameter, Ray, Solver, Space, System, ThinLens


def make_solver():
    distance = Parameter(0.5).variable(bounds=(0, 1))
    solver = Solver(System(Space(d=distance), source=Ray(x=0, theta=0, n=1)))
    solver.target(distance - 0.25)
    return solver


@pytest.mark.parametrize("value,error", [
    (True, TypeError), (1j, TypeError), ("0.1", TypeError), (object(), TypeError),
    (-1, ValueError), (np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError),
])
def test_invalid_feasibility_tolerance_fails_before_compilation(value, error, monkeypatch):
    solver = make_solver()
    monkeypatch.setattr(solver, "_finalize_constraints", lambda: pytest.fail("Compilation must follow validation"))
    with pytest.raises(error, match="feasibility_tolerance"):
        solver.solve(feasibility_tolerance=value)


@pytest.mark.parametrize("value", [None, 1, 0, "yes", 1.0])
def test_jit_must_be_a_bool_and_is_validated_before_compilation(value, monkeypatch):
    solver = make_solver()
    monkeypatch.setattr(solver, "_finalize_constraints", lambda: pytest.fail("Compilation must follow validation"))
    with pytest.raises(TypeError, match="jit must be a bool"):
        solver.solve(jit=value)


@pytest.mark.parametrize("collection", ["objectives", "constraints"])
def test_invalid_model_entries_raise_type_error(collection):
    solver = make_solver()
    getattr(solver, "_" + collection).append(object())
    with pytest.raises(TypeError, match=collection):
        solver.solve()


def test_empty_optimization_model_is_rejected():
    solver = Solver(System(source=Ray(x=Parameter(1).variable(), theta=0, n=1)))
    with pytest.raises(ValueError, match="at least one objective or constraint"):
        solver.solve()


def test_model_without_variable_parameters_is_rejected():
    solver = Solver(System(source=Ray(x=0, theta=0, n=1)))
    solver.target(lambda result: result.final.x)
    with pytest.raises(ValueError, match="at least one variable"):
        solver.solve()


@pytest.mark.parametrize("arguments", [
    {"feasibility_tolerance": 0},
    {"feasibility_tolerance": 1e-9},
    {"x0": [0], "x_lower": [0], "x_upper": [1]},
    {"x0": [1], "x_lower": [0], "x_upper": [1]},
    {"x0": [0.5], "x_lower": [0.5], "x_upper": [0.5]},
    {"x0": [2], "x_lower": [-np.inf], "x_upper": [np.inf]},
])
def test_valid_inputs_reach_preparation_without_mutating_defaults(arguments, monkeypatch):
    solver = make_solver()
    values = solver.simulation.initial_values.values.copy()
    metadata = solver.simulation.parameter_data
    captured = []
    finalize = solver._finalize_constraints

    def record_preparation():
        captured.append(True)
        return finalize()

    monkeypatch.setattr(solver, "_finalize_constraints", record_preparation)
    from tests.helpers.backend import capture_problem
    capture_problem(solver, **arguments)
    assert captured == [True]
    np.testing.assert_array_equal(solver.simulation.initial_values.values, values)
    assert solver.simulation.parameter_data == metadata


# ----------------------------
# ERROR HINTS
# ----------------------------
def _branching_solver():
    distance = Parameter(0.5).variable(bounds=(0, 1))
    simulation = System(Space(d=distance), source=Ray(x=1e-3, theta=1e-3, n=1)).build()

    def residual(sim):
        value = sim.final.x
        return value if value > 0 else -value  # Python control flow on a runtime value

    return Solver(simulation).target(residual)


def test_a_branching_callable_fails_under_jit_with_a_hint_to_pass_jit_false_and_works_without_jit():
    from diffractix.backend import current_backend
    if current_backend() != "jax":
        pytest.skip("Only JAX traces callables abstractly.")
    with pytest.raises(TypeError, match=r"jit=False") as error:
        _branching_solver().solve(jit=True)
    assert error.value.__cause__ is not None
    assert _branching_solver().solve(jit=False).success


def test_without_jit_a_callable_that_needs_concrete_values_fails_unchanged():
    from diffractix.backend import current_backend, tracing_errors
    if current_backend() != "jax":
        pytest.skip("Only JAX traces callables abstractly.")
    distance = Parameter(0.5).variable(bounds=(0, 1))
    simulation = System(Space(d=distance), source=Ray(x=1e-3, theta=1e-3, n=1)).build()

    def residual(sim):
        value = sim.final.x
        if value > 0:                     # deferred: branches on a result value
            return np.asarray(value)      # NumPy cannot convert a value JAX is differentiating
        return -value

    solver = Solver(simulation).target(residual)
    with pytest.raises(tracing_errors()) as error:
        solver.solve(jit=False)
    assert "jit=False" not in str(error.value)


def test_chained_comparisons_explain_how_to_write_the_two_bounds():
    distance = Parameter(0.5)
    with pytest.raises(TypeError, match="chained comparison.*solver.require"):
        _ = 0.1 < distance < 0.9


@pytest.mark.parametrize("make_residual", [
    lambda xp: (lambda sim: xp.abs(sim.final.x)),
    lambda xp: (lambda sim: xp.where(sim.final.x > 0, sim.final.x, -sim.final.x)),
    lambda xp: (lambda sim: np.abs(sim.final.x)),
    lambda xp: (lambda sim: np.maximum(sim.final.x, -sim.final.x)),
], ids=["xp-abs", "xp-where", "np-abs", "np-maximum"])
def test_callables_using_backend_functions_solve_on_every_backend(make_residual):
    from diffractix.backend import xp
    distance = Parameter(0.05).variable(bounds=(0, 1))
    simulation = System(ThinLens(f=0.1), Space(d=distance), source=Ray(x=1e-3, theta=0, n=1)).build()
    solver = Solver(simulation).target(value=make_residual(xp), target=0.0, scale=1e-3)
    assert solver.solve().value(distance) == pytest.approx(0.1, abs=1e-6)


def test_callables_using_numpy_functions_stay_traceable():
    distance = Parameter(0.05).variable(bounds=(0, 1))
    simulation = System(ThinLens(f=0.1), Space(d=distance), source=Ray(x=1e-3, theta=0, n=1)).build()
    solver = Solver(simulation).target(lambda sim: np.sqrt(sim.final.x ** 2) / 1e-3)
    cache = solver._build_cache(False, simulation.initial_values.values)
    assert cache.callable_objectives == () and len(cache.objective_descriptors) == 1
