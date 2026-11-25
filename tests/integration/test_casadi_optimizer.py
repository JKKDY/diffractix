"""The CasADi optimizer keeps one compiled NLP across repeated solves."""

import numpy as np
import pytest

from diffractix import Optimizer, Parameter, Ray, RayBundle, Solver, Space, System, ThinLens
from diffractix.backend import current_backend, xp


def test_casadi_optimizer_requires_casadi_numerical_backend():
    xp.asarray(0)
    if current_backend() == "casadi":
        pytest.skip("Requires a numerical backend other than CasADi.")
    x = Parameter(0.5).variable(bounds=(0, 2))
    solver = Solver(System(source=Ray(x=x, theta=0, n=1))).target(x - 1)
    with pytest.raises(ValueError, match="requires the CasADi numerical backend"):
        solver.solve(optimizer=Optimizer.CASADI)


@pytest.fixture
def model():
    xp.asarray(0)
    if current_backend() != "casadi":
        pytest.skip("Requires the CasADi numerical backend.")
    x = Parameter(1.0).variable(bounds=(0, 2))
    theta = Parameter(0.8).variable(bounds=(0, 2))
    return Solver(System(source=Ray(x=x, theta=theta, n=1))), x, theta


def test_casadi_optimizer_reuses_the_nlp_when_pin_and_active_rows_change(model, monkeypatch):
    from unittest.mock import Mock

    from diffractix.solver.optimizers import casadi

    compile_nlp = Mock(wraps=casadi._compile_nlp)
    monkeypatch.setattr(casadi, "_compile_nlp", compile_nlp)
    solver, x, theta = model
    solver.target(x - 1.8)
    solver.require(x <= 1.5, theta == 0.5)

    pinned = solver.solve(optimizer="casadi", x_lower=[1.0, 0], x_upper=[1.0, 2])
    assert solver._cache.active_rows.size == 1
    released = solver.solve(optimizer="casadi", x0=pinned.x)
    assert solver._cache.active_rows.size == 2
    assert compile_nlp.call_count == 1

    assert pinned.success and pinned.feasible, pinned.message
    assert released.success and released.feasible, released.message
    assert pinned.value(x) == pytest.approx(1.0)
    assert released.value(x) == pytest.approx(1.5)
    assert released.value(theta) == pytest.approx(0.5)
    assert released.iterations is not None
    assert "sqpmethod" in released.message


def test_casadi_optimizer_returns_an_unsuccessful_solution_at_its_iteration_limit(model):
    solver, x, _ = model
    solver.target(x - 1.8)
    result = solver.solve(optimizer=Optimizer.CASADI, options={"max_iter": 0})
    assert not result.success
    assert result.value(x) == pytest.approx(1.0)
    assert result.iterations == 0
    assert "Maximum_Iterations_Exceeded" in result.message


@pytest.mark.parametrize("case", ["stationary", "nonstationary", "infeasible"])
def test_small_sqp_step_is_successful_only_when_kkt_tolerances_are_met(model, case):
    solver, x, theta = model
    if case == "stationary":
        solver.target(x - 1.8)
        solver.require(theta == 0.5)
        x0, options = [1.8, 0.5], {"tol_pr": 0.0}
    elif case == "nonstationary":
        solver.target(x**2 - 2)
        solver.require(theta == 0.5)
        x0, options = [1.0, 0.5], {"min_step_size": 1.0}
    else:
        solver.target(x - 1)
        solver.require(solver.simulation.symbolic.initial.theta**2 == 0.25)
        x0, options = [1.0, 0.8], {"min_step_size": 1.0}

    result = solver.solve(optimizer="casadi", x0=x0, options=options)
    assert result._result.raw["return_status"] == "Search_Direction_Becomes_Too_Small"
    assert result.success is (case == "stationary")
    assert result.feasible is (case != "infeasible")
    assert ("KKT tolerances met" in result.message) is (case == "stationary")


def test_casadi_optimizer_solves_a_nonlinear_model_without_constraints(model):
    length, radius = 0.1, 0.001
    focal = Parameter(0.15).variable(bounds=(0.05, 0.3))
    source = RayBundle(x=np.array([-radius, 0, radius]), theta=np.zeros(3), n=1)
    solver = Solver(System(ThinLens(f=focal), Space(d=length), source=source))
    solver.target(lambda sim: sim.final.x / radius)
    for _ in range(5):
        solution = solver.solve(optimizer="casadi")
        assert solution.success and solution.feasible, solution.message
        assert solution.value(focal) == pytest.approx(length, abs=1e-7)


def test_casadi_optimizer_solves_a_ray_focus_with_negative_initial_curvature(model):
    focal = Parameter(0.2).variable(bounds=(0.05, 0.5))
    source = Ray(x=0.01, theta=0, n=1)
    simulation = System(ThinLens(f=focal), Space(d=0.1), source=source).build()
    solver = Solver(simulation).target(value=simulation.symbolic.final.x, target=0.0, scale=0.01)
    solution = solver.solve(optimizer="casadi")
    assert solution.success and solution.feasible, solution.message
    assert solution.value(focal) == pytest.approx(0.1, abs=1e-7)
