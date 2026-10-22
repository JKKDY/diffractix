"""Input validation and fixed Simulation ownership for reusable Solver callbacks."""

from unittest.mock import patch

import numpy as np
import pytest

from diffractix import Parameter, Solver
from diffractix.solver import solver as solver_module
from tests.helpers.backend import capture_problem
from tests.helpers.simulation import make_simulation


@pytest.fixture
def solver():
    parameter = Parameter(1.5).variable(bounds=(1.0, 4.0))
    return Solver(make_simulation((parameter,))).target(parameter - 0.2)


@pytest.mark.parametrize("argument", ["x0", "x_lower", "x_upper"])
@pytest.mark.parametrize("value", [1.5, [[1.5]], [1.0, 2.0]])
def test_override_arrays_require_canonical_vector_shape(solver, argument, value):
    with patch.object(solver, "_build_cache") as build, patch.object(
        solver, "_reduce_constraints",
    ) as reduce, patch.object(solver_module, "solve_scipy") as dispatch:
        with pytest.raises(ValueError, match=rf"{argument}.*shape"):
            solver.solve(**{argument: value})
    build.assert_not_called()
    reduce.assert_not_called()
    dispatch.assert_not_called()
    assert solver._cache is None


@pytest.mark.parametrize("overrides, message", [
    ({"x0": [np.nan]}, "x0.*finite"),
    ({"x0": [np.inf]}, "x0.*finite"),
    ({"x0": [-np.inf]}, "x0.*finite"),
    ({"x_lower": [np.nan]}, "bounds.*NaN"),
    ({"x_upper": [np.nan]}, "bounds.*NaN"),
    ({"x_lower": [3.0], "x_upper": [2.0]}, "x_lower.*x_upper"),
    ({"x0": [0.5]}, "x0.*within"),
    ({"x0": [4.5]}, "x0.*within"),
    ({"x_lower": [2.0]}, "x0.*within"),
    ({"x_upper": [1.0]}, "x0.*within"),
])
def test_invalid_overrides_fail_before_compilation_reduction_or_dispatch(solver, overrides, message):
    with patch.object(solver, "_build_cache") as build, patch.object(
        solver, "_reduce_constraints",
    ) as reduce, patch.object(solver_module, "solve_scipy") as dispatch:
        with pytest.raises(ValueError, match=message):
            solver.solve(**overrides)
    build.assert_not_called()
    reduce.assert_not_called()
    dispatch.assert_not_called()
    assert solver._cache is None


@pytest.mark.parametrize("start, lower, upper", [
    (1.5, -np.inf, np.inf),
    (1.5, -np.inf, 4.0),
    (1.5, 1.0, np.inf),
    (1.0, 1.0, 4.0),
    (4.0, 1.0, 4.0),
    (1.5, 1.5, 1.5),
])
def test_overrides_accept_infinite_and_inclusive_bounds(solver, start, lower, upper):
    problem = capture_problem(solver, x0=[start], x_lower=[lower], x_upper=[upper])
    np.testing.assert_array_equal(problem.x0, [start])
    np.testing.assert_array_equal(problem.x_lower, [lower])
    np.testing.assert_array_equal(problem.x_upper, [upper])
    np.testing.assert_array_equal(solver.simulation.initial_values, [1.5])
    np.testing.assert_array_equal(solver._parameter_bounds(), [[1.0], [4.0]])


def test_invalid_later_solve_preserves_cache_and_next_solve_defaults(solver):
    capture_problem(solver)
    cache = solver._cache
    with pytest.raises(ValueError, match="x0.*within"):
        solver.solve(x_lower=[2.0])
    assert solver._cache is cache
    problem = capture_problem(solver, x0=[2.0])
    assert solver._cache is cache
    np.testing.assert_array_equal(problem.x0, [2.0])
    np.testing.assert_array_equal(problem.x_lower, [1.0])
    np.testing.assert_array_equal(problem.x_upper, [4.0])


def test_simulation_cannot_be_reassigned_even_before_first_solve(solver):
    original = solver.simulation
    other = make_simulation((Parameter(2.0).variable(),))
    for replacement in (original, other):
        with pytest.raises(AttributeError, match="cannot be reassigned"):
            solver.simulation = replacement
        assert solver.simulation is original
