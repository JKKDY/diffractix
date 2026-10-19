"""Objective shorthand and constraint normalization."""

import math
import numpy as np
import pytest
from diffractix.graph import Literal, Parameter
from diffractix.solver.objective import Objective
from diffractix.solver.constraint import normalize_to_constraint
from tests.helpers.solver import make_solver


@pytest.mark.parametrize("args, kwargs, error, message", [
    ((), {"value": lambda ctx: 1.0}, ValueError, "target must be explicitly"),
    ((), {"value": 1.0, "target": 0}, TypeError, "value must be a callable"),
    ((), {"value": lambda ctx: 1.0, "target": 0, "scale": 0}, ValueError, "scale must be nonzero"),
    ((lambda ctx: 1.0,), {"value": lambda ctx: 1.0, "target": 0}, ValueError, "positional objectives"),
    ((lambda ctx: 1.0,), {"target": 0}, ValueError, "require value"),
    ((lambda ctx: 1.0,), {"scale": 2}, ValueError, "require value"),
    ((), {"label": "ambiguous"}, ValueError, "single residual"),
    ((lambda ctx: 1.0, lambda ctx: 2.0), {"label": "ambiguous"}, ValueError, "single residual"),
    ((Objective(lambda ctx: 1.0),), {"label": "override"}, ValueError, "Objective instances"),
])
def test_target_rejects_ambiguous_or_invalid_shorthand(args, kwargs, error, message):
    solver, _, _ = make_solver()
    with pytest.raises(error, match=message):
        solver.target(*args, **kwargs)
    assert solver.objectives == []


@pytest.mark.parametrize(
    ("make_comparison", "lower_bound", "upper_bound"),
    (
        (
            lambda parameter: parameter > 2.0,
            math.nextafter(2.0, math.inf),
            np.inf,
        ),
        (
            lambda parameter: parameter < 2.0,
            -np.inf,
            math.nextafter(2.0, -math.inf),
        ),
        (
            lambda parameter: Literal(2.0) < parameter,
            math.nextafter(2.0, math.inf),
            np.inf,
        ),
        (
            lambda parameter: Literal(2.0) > parameter,
            -np.inf,
            math.nextafter(2.0, -math.inf),
        ),
    ),
)
def test_strict_comparisons_normalize_to_open_floating_point_bounds(
    make_comparison,
    lower_bound,
    upper_bound,
):
    parameter = Parameter(3.0)

    constraint = normalize_to_constraint(make_comparison(parameter))

    assert constraint.evaluate is parameter
    assert constraint.lower_bound == lower_bound
    assert constraint.upper_bound == upper_bound


@pytest.mark.parametrize(
    ("make_comparison", "lower_bound", "upper_bound"),
    (
        (
            lambda left, right: left > right,
            math.nextafter(0.0, math.inf),
            np.inf,
        ),
        (
            lambda left, right: left < right,
            -np.inf,
            math.nextafter(0.0, -math.inf),
        ),
    ),
)
def test_strict_parameter_relations_normalize_to_open_residual_bounds(
    make_comparison,
    lower_bound,
    upper_bound,
):
    left = Parameter(3.0)
    right = Parameter(2.0)

    constraint = normalize_to_constraint(make_comparison(left, right))

    assert constraint.evaluate is not left
    assert constraint.evaluate is not right
    assert constraint.lower_bound == lower_bound
    assert constraint.upper_bound == upper_bound


def test_empty_objective_and_constraint_collections_compile_to_empty_tuples():
    solver, _, _ = make_solver()

    assert solver._compile_objectives() == ()
    assert solver._compile_constraints(()) == ()
