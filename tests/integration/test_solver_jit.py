"""JIT and dynamic numerical fallback through actual optical solves."""
import numpy as np
import pytest
from diffractix import Constraint, Parameter, Ray, Solver, Space, System


@pytest.mark.parametrize("use_jit", [False, True])
def test_vector_optical_constraints_and_scalar_target_solve_together(use_jit):
    position = Parameter(0.2).variable(bounds=(-2, 2))
    angle = Parameter(0.1).variable(bounds=(-1, 1))
    solver = Solver(System(Space(d=1), optical_input=Ray(x=position, theta=angle, n=1)))
    solver.target(lambda sim: sim.final.x - 1)
    solver.require(Constraint(lambda sim: sim.final.vector, lower_bound=-0.25, upper_bound=0.4))
    solution = solver.solve(jit=use_jit, options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.run().final.x == pytest.approx(0.4, abs=1e-7)
    assert solution.cost == pytest.approx(0.36, abs=1e-7)
    vector = [entry for entry in solution.constraints if entry.value.shape == (2,)][0]
    assert vector.satisfied
    np.testing.assert_allclose(vector.value, solution.run().final.vector)


def test_dynamic_branch_optimizes_numerical_outputs_eagerly():
    position = Parameter(0.5).variable(bounds=(0.1, 2))
    solver = Solver(System(optical_input=Ray(x=position, theta=0, n=1)))

    def residual(sim):
        if sim.final.x > 0:
            return sim.final.x ** 2 - 1
        return 10.

    solver.target(residual)
    solution = solver.solve(jit=False, options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(position) == pytest.approx(1, abs=1e-6)
    assert solution.objectives[0].cost < 1e-11


def test_deferred_returned_node_fails_instead_of_becoming_a_live_graph():
    position = Parameter(0.5).variable(bounds=(0.1, 2))
    solver = Solver(System(optical_input=Ray(x=position, theta=0, n=1)))

    def residual(sim):
        return position - 1 if sim.final.x > 0 else 10.

    solver.target(residual)
    with pytest.raises(TypeError, match="numerical.*symbolic Node"):
        solver.solve(jit=False)


def test_user_failure_is_preserved_and_preparation_can_be_retried():
    position = Parameter(0.5).variable(bounds=(0.1, 2))
    solver = Solver(System(optical_input=Ray(x=position, theta=0, n=1)))
    fail = [True]

    def residual(sim):
        if fail[0]:
            raise RuntimeError("user's optical objective failed")
        return sim.final.x - 1

    solver.target(residual)
    with pytest.raises(RuntimeError, match="user's optical objective"):
        solver.solve()
    assert solver._cache is None
    fail[0] = False
    solution = solver.solve()
    assert solution.success and solution.feasible
    assert solution.value(position) == pytest.approx(1, abs=1e-7)
