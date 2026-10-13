"""Backend-neutral optimization Problem validation."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from tests.helpers.optimization import make_constrained_problem, make_problem


def test_problem_dimensions_and_bound_length_validation():
    problem = make_problem()
    assert (problem.n_variables, problem.n_constraints) == (1, 0)
    constrained = make_constrained_problem()
    assert (constrained.n_variables, constrained.n_constraints) == (2, 4)

    with pytest.raises(ValueError, match="Variable bounds"):
        replace(problem, x_upper=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Constraint bounds"):
        replace(problem, constraint_upper=np.array([1.0]))
