from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import autograd.numpy as np

from diffractix.solver import Backend, Solver
from diffractix.solver.problem import Problem


def make_problem():
    return Problem(
        x0=np.array([1.0]),
        x_lower=np.array([0.0]),
        x_upper=np.array([2.0]),
        objective=lambda x: x[0] ** 2,
        gradient=lambda x: np.array([2 * x[0]]),
        constraints=lambda x: np.array([]),
        jacobian=lambda x: np.empty((0, 1)),
        constraint_lower=np.array([]),
        constraint_upper=np.array([]),
    )


def test_scipy_options_are_forwarded(monkeypatch):
    from diffractix.solver.backends import scipy as scipy_backend

    received = {}

    def minimize(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            x=np.array([0.5]),
            success=True,
            fun=0.25,
            message="done",
            nit=3,
        )

    monkeypatch.setattr(scipy_backend, "minimize", minimize)
    options = {"maxiter": 500, "verbose": 2}

    scipy_backend.solve_scipy(make_problem(), options=options)

    assert received["method"] == "trust-constr"
    assert received["options"] == options
    assert received["options"] is not options


def test_scipy_none_options_becomes_empty_mapping(monkeypatch):
    from diffractix.solver.backends import scipy as scipy_backend

    received = {}

    def minimize(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            x=np.array([0.5]), success=True, fun=0.25, message="done"
        )

    monkeypatch.setattr(scipy_backend, "minimize", minimize)

    scipy_backend.solve_scipy(make_problem())

    assert received["options"] == {}


def test_ipopt_options_include_default_and_allow_override(monkeypatch):
    fake_cyipopt = ModuleType("cyipopt")
    captured = {}

    class FakeProblem:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            captured["options"] = []

        def add_option(self, name, value):
            captured["options"].append((name, value))

        def solve(self, x0):
            return x0, {"status": 0, "obj_val": 0.0, "status_msg": "done"}

    fake_cyipopt.Problem = FakeProblem
    monkeypatch.setitem(sys.modules, "cyipopt", fake_cyipopt)
    sys.modules.pop("diffractix.solver.backends.ipopt", None)
    ipopt_backend = importlib.import_module("diffractix.solver.backends.ipopt")

    options = {"tol": 1e-9, "hessian_approximation": "exact"}
    ipopt_backend.solve_ipopt(make_problem(), options=options)

    assert captured["options"] == [
        ("hessian_approximation", "exact"),
        ("tol", 1e-9),
    ]
    assert options == {"tol": 1e-9, "hessian_approximation": "exact"}


def test_ipopt_none_options_applies_limited_memory_default(monkeypatch):
    fake_cyipopt = ModuleType("cyipopt")
    captured = []

    class FakeProblem:
        def __init__(self, **kwargs):
            pass

        def add_option(self, name, value):
            captured.append((name, value))

        def solve(self, x0):
            return x0, {"status": 0, "obj_val": 0.0, "status_msg": "done"}

    fake_cyipopt.Problem = FakeProblem
    monkeypatch.setitem(sys.modules, "cyipopt", fake_cyipopt)
    sys.modules.pop("diffractix.solver.backends.ipopt", None)
    ipopt_backend = importlib.import_module("diffractix.solver.backends.ipopt")

    ipopt_backend.solve_ipopt(make_problem())

    assert captured == [("hessian_approximation", "limited-memory")]


def test_lazy_backend_entry_point_forwards_options(monkeypatch):
    from diffractix.solver.backends import scipy as scipy_backend
    from diffractix.solver import backends

    received = {}

    def solve(problem, method=None, options=None):
        received.update(problem=problem, method=method, options=options)
        return "result"

    monkeypatch.setattr(scipy_backend, "solve_scipy", solve)
    options = {"gtol": 1e-8}

    assert backends.solve_scipy("problem", method="SLSQP", options=options) == "result"
    assert received == {"problem": "problem", "method": "SLSQP", "options": options}


def test_solver_forwards_options_without_mutating_callers_mapping(monkeypatch):
    from diffractix.solver import solver as solver_module

    solver = object.__new__(Solver)
    solver.objectives = ()
    solver.constraints = ()
    solver.simulation = SimpleNamespace(initial_values=np.array([1.0]), run=lambda theta: None)
    solver._compile_objectives = lambda: ()
    solver._compile_constraints = lambda: ()
    solver._parameter_bounds = lambda: (np.array([0.0]), np.array([2.0]))
    solver._constraint_bounds = lambda constraints: (np.array([]), np.array([]))

    received = {}

    def solve_scipy(problem, method=None, options=None):
        received.update(problem=problem, method=method, options=options)
        return "result"

    monkeypatch.setattr(solver_module, "solve_scipy", solve_scipy)
    options = {"maxiter": 100}

    assert solver.solve(Backend.SCIPY, method="SLSQP", options=options) == "result"
    assert received["method"] == "SLSQP"
    assert received["options"] == options
    assert received["options"] is not options
    assert options == {"maxiter": 100}
