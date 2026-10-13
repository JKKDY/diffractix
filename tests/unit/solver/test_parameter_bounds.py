"""Canonical solver variable bounds from frozen metadata."""

from types import SimpleNamespace
import numpy as numpy
from diffractix.graph import Parameter
from tests.helpers.solver import make_solver, make_solver_with_reordered_parameters


def test_parameter_bounds_follow_canonical_theta_order():
    solver, _, _, _ = make_solver_with_reordered_parameters()

    lower_bounds, upper_bounds = solver._parameter_bounds()

    numpy.testing.assert_array_equal(lower_bounds, [-4.0, -2.0])
    numpy.testing.assert_array_equal(upper_bounds, [4.0, 2.0])


def test_parameter_bounds_exclude_fixed_metadata():
    solver, _, parameter = make_solver()
    fixed = Parameter(8.0, name="fixed")
    solver.simulation.parameter_info[id(fixed)] = SimpleNamespace(
        value=8.0,
        parameter_index=None,
        lower_bound=7.0,
        upper_bound=9.0,
    )

    lower_bounds, upper_bounds = solver._parameter_bounds()

    numpy.testing.assert_array_equal(lower_bounds, [-10.0])
    numpy.testing.assert_array_equal(upper_bounds, [10.0])
    assert id(parameter) in solver.simulation.parameter_info


def test_parameter_bounds_are_empty_without_variable_metadata():
    solver, _, _ = make_solver()
    fixed = Parameter(8.0, name="fixed")
    solver.simulation.parameter_info = {
        id(fixed): SimpleNamespace(
            value=8.0,
            parameter_index=None,
            lower_bound=7.0,
            upper_bound=9.0,
        ),
    }

    lower_bounds, upper_bounds = solver._parameter_bounds()

    assert lower_bounds.shape == (0,)
    assert upper_bounds.shape == (0,)
