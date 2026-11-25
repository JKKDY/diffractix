"""A changing reference gives the same scaled preference across optimizers."""

import numpy as np
import pytest

from diffractix import Parameter, Ray, Solver, System
from diffractix.backend import current_backend, xp


@pytest.fixture(params=["nlopt", "scipy", "casadi"])
def optimizer(request):
    name = request.param
    if name == "casadi":
        xp.asarray(0)
        if current_backend() != "casadi":
            pytest.skip("Requires the CasADi numerical backend.")
    else:
        pytest.importorskip(name)
    return name


def test_proximal_projects_reference_in_scaled_coordinates(optimizer):
    a = Parameter(1).variable(bounds=(0, 10))
    b = Parameter(8).variable(bounds=(0, 20))
    solver = Solver(System(source=Ray(x=a, theta=b, n=1))).require(a + b == 10)
    reference = solver.simulation.initial_values
    reference[a], reference[b] = 2, 6

    plain = solver.solve(optimizer=optimizer)
    near = solver.solve(optimizer=optimizer, proximal=reference, proximal_weight=0.5)

    assert plain.feasible
    assert plain.proximal_cost == 0
    assert near.success and near.feasible, near.message
    # Of the missing 2, the closest scaled point assigns 2*25/(25+100) to a.
    np.testing.assert_allclose(near.x, [2.4, 7.6], atol=1e-7)
    assert near.cost == 0
    assert near.proximal_cost == pytest.approx(0.5 * (0.4**2 / 25 + 1.6**2 / 100))
    assert near.objectives == ()


def test_proximal_does_not_move_the_unique_feasible_point_or_count_a_pin(optimizer):
    a = Parameter(1).variable(bounds=(0, 4))
    b = Parameter(2).variable(bounds=(0, 4))
    solver = Solver(System(source=Ray(x=a, theta=b, n=1))).require(a == 3)
    bounds = {"x_lower": [0, 2], "x_upper": [4, 2]}
    plain = solver.solve(optimizer=optimizer, **bounds)
    near = solver.solve(optimizer=optimizer, proximal=[1, 100], **bounds)

    assert plain.feasible and near.feasible
    np.testing.assert_allclose(near.x, plain.x, atol=1e-7)
    np.testing.assert_allclose(near.x, [3, 2], atol=1e-7)
    assert near.proximal_cost == pytest.approx(1)


def test_proximal_weight_changes_the_soft_optimum_and_costs_are_separate(optimizer):
    a = Parameter(1).variable(bounds=(0, 2))
    solver = Solver(System(source=Ray(x=a, theta=0, n=1))).target(a - 1.5)
    disabled = solver.solve(optimizer=optimizer)
    near = solver.solve(optimizer=optimizer, proximal=[0.5], proximal_weight=1)

    assert disabled.value(a) == pytest.approx(1.5)
    assert disabled.proximal_cost == 0
    assert near.success and near.value(a) == pytest.approx(1)
    assert "objectives" not in vars(near)
    assert near.cost == pytest.approx(0.25)
    assert vars(near)["objectives"][0].cost == pytest.approx(0.25)
    assert near.proximal_cost == pytest.approx(0.25)


def test_proximal_can_be_the_only_objective(optimizer):
    a = Parameter(1).variable(bounds=(0, 2))
    solver = Solver(System(source=Ray(x=a, theta=0, n=1)))
    solution = solver.solve(optimizer=optimizer, proximal=[3])
    assert solution.success and solution.feasible, solution.message
    assert solution.value(a) == pytest.approx(2)
    assert solution.cost == 0
    assert solution.proximal_cost == pytest.approx(1)


@pytest.mark.parametrize("kwargs, error, message", [
    ({"proximal": [1, 2]}, ValueError, "proximal must have shape"),
    ({"proximal": [np.inf]}, ValueError, "finite real"),
    ({"proximal": [1j]}, ValueError, "finite real"),
    ({"proximal_weight": -1}, ValueError, "finite and non-negative"),
    ({"proximal": [1], "proximal_weight": -1}, ValueError, "finite and non-negative"),
    ({"proximal": [1], "proximal_weight": 0}, ValueError, "must be positive"),
    ({"proximal_weight": True}, TypeError, "real numeric scalar"),
])
def test_proximal_rejects_invalid_inputs(kwargs, error, message):
    a = Parameter(1).variable(bounds=(0, 2))
    solver = Solver(System(source=Ray(x=a, theta=0, n=1))).target(a - 0.5)
    with pytest.raises(error, match=message):
        solver.solve(**kwargs)
    solver.target(a - 1.5)


def test_casadi_reuses_the_nlp_with_new_proximal_references_and_weights(monkeypatch):
    xp.asarray(0)
    if current_backend() != "casadi":
        pytest.skip("Requires the CasADi numerical backend.")
    from unittest.mock import Mock

    from diffractix.solver.optimizers import casadi

    compile_nlp = Mock(wraps=casadi._compile_nlp)
    monkeypatch.setattr(casadi, "_compile_nlp", compile_nlp)
    a = Parameter(1).variable(bounds=(0, 2))
    solver = Solver(System(source=Ray(x=a, theta=0, n=1))).target(a - 1.5)

    plain = solver.solve(optimizer="casadi")
    first = solver.solve(optimizer="casadi", proximal=[0.5])
    second = solver.solve(optimizer="casadi", proximal=[0], proximal_weight=2)
    assert compile_nlp.call_count == 1
    assert plain.value(a) == pytest.approx(1.5)
    assert first.value(a) == pytest.approx(1)
    assert second.value(a) == pytest.approx(0.5)
