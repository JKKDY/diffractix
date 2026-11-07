"""Validate solve inputs before compiling the optimization model."""

import numpy as np
import pytest

from diffractix import Parameter, Ray, Solver, Space, System


def make_solver():
    distance = Parameter(0.5).variable(lower_bound=0, upper_bound=1)
    solver = Solver(System(Space(d=distance), optical_input=Ray(x=0, theta=0, n=1)))
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


@pytest.mark.parametrize("arguments,message", [
    ({"x0": []}, "x0.*shape"),
    ({"x0": [[0.5]]}, "x0.*shape"),
    ({"x_lower": []}, "x_lower.*shape"),
    ({"x_upper": [[1]]}, "x_upper.*shape"),
    ({"x0": [np.nan]}, "x0.*finite"),
    ({"x0": [np.inf]}, "x0.*finite"),
    ({"x0": [-np.inf]}, "x0.*finite"),
    ({"x_lower": [np.nan]}, "bounds.*NaN"),
    ({"x_upper": [np.nan]}, "bounds.*NaN"),
    ({"x_lower": [0.9], "x_upper": [0.1]}, "x_lower.*x_upper"),
    ({"x0": [-0.1]}, "x0.*within"),
    ({"x0": [1.1]}, "x0.*within"),
    ({"x_lower": [0.75]}, "x0.*within"),
    ({"x_upper": [0.25]}, "x0.*within"),
])
def test_invalid_initial_values_and_bounds_fail_before_compilation(arguments, message, monkeypatch):
    solver = make_solver()
    monkeypatch.setattr(solver, "_finalize_constraints", lambda: pytest.fail("Compilation must follow validation"))
    with pytest.raises(ValueError, match=message):
        solver.solve(**arguments)


@pytest.mark.parametrize("collection", ["objectives", "constraints"])
def test_invalid_model_entries_raise_type_error(collection):
    solver = make_solver()
    getattr(solver, collection).append(object())
    with pytest.raises(TypeError, match=collection):
        solver.solve()


def test_empty_optimization_model_is_rejected():
    solver = Solver(System(optical_input=Ray(x=Parameter(1).variable(), theta=0, n=1)))
    with pytest.raises(ValueError, match="at least one objective or constraint"):
        solver.solve()


def test_model_without_variable_parameters_is_rejected():
    solver = Solver(System(optical_input=Ray(x=0, theta=0, n=1)))
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
    solver.solve(**arguments)
    assert captured == [True]
    np.testing.assert_array_equal(solver.simulation.initial_values.values, values)
    assert solver.simulation.parameter_data == metadata
