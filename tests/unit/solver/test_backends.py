"""Backend adapters, lazy imports, and dispatch using mocked optimizers."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from diffractix.solver import Backend, Objective, OptimizationResult, Solution, Solver
from diffractix.solver.problem import Problem
from tests.helpers.optimization import make_constrained_problem, make_problem, make_scaling_problem


def import_fake_cyipopt_backend(monkeypatch):
    fake_cyipopt = ModuleType("cyipopt")
    solvers = []

    class FakeProblem:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.options = []
            self.solution = np.asarray(kwargs["lb"], dtype=float)
            self.info = {
                "status": 0,
                "obj_val": 0.25,
                "status_msg": b"solved",
            }
            solvers.append(self)

        def add_option(self, name, value):
            self.options.append((name, value))

        def solve(self, x0):
            self.x0 = np.asarray(x0)
            return self.solution, self.info

    fake_cyipopt.Problem = FakeProblem
    monkeypatch.setitem(sys.modules, "cyipopt", fake_cyipopt)

    backend = importlib.import_module("diffractix.solver.backends.ipopt")
    monkeypatch.setattr(backend, "cyipopt", fake_cyipopt)
    return backend, fake_cyipopt, solvers


def import_fake_nlopt_backend(monkeypatch):
    fake_nlopt = ModuleType("nlopt")
    fake_nlopt.SUCCESS = 1
    fake_nlopt.STOPVAL_REACHED = 2
    fake_nlopt.FTOL_REACHED = 3
    fake_nlopt.XTOL_REACHED = 4
    fake_nlopt.MAXEVAL_REACHED = 5
    fake_nlopt.MAXTIME_REACHED = 6
    fake_nlopt.LD_SLSQP = 40
    fake_nlopt.LN_COBYLA = 41
    fake_nlopt.LD_MMA = 42
    fake_nlopt.next_status = fake_nlopt.XTOL_REACHED
    fake_nlopt.supported_params = {"inner_maxeval"}
    optimizers = []

    class FakeOpt:
        def __init__(self, algorithm, n_variables):
            self.algorithm = algorithm
            self.n_variables = n_variables
            self.lower_bounds = None
            self.upper_bounds = None
            self.objective = None
            self.inequality = None
            self.equality = None
            self.options = {}
            self.params = {}
            optimizers.append(self)

        def set_lower_bounds(self, value):
            self.lower_bounds = np.asarray(value)

        def set_upper_bounds(self, value):
            self.upper_bounds = np.asarray(value)

        def set_min_objective(self, callback):
            self.objective = callback

        def add_inequality_mconstraint(self, callback, tolerance):
            self.inequality = (callback, np.asarray(tolerance))

        def add_equality_mconstraint(self, callback, tolerance):
            self.equality = (callback, np.asarray(tolerance))

        def has_param(self, name):
            return name in fake_nlopt.supported_params

        def set_param(self, name, value):
            self.params[name] = value

        def optimize(self, x0):
            self.x0 = np.asarray(x0)
            return self.x0 * 0.5

        def last_optimize_result(self):
            return fake_nlopt.next_status

        def last_optimum_value(self):
            return 0.25

        def get_algorithm_name(self):
            return "fake algorithm"

        def get_numevals(self):
            return 7

    for option in (
        "stopval",
        "ftol_rel",
        "ftol_abs",
        "xtol_rel",
        "xtol_abs",
        "x_weights",
        "maxeval",
        "maxtime",
        "initial_step",
        "population",
        "vector_storage",
    ):
        setattr(
            FakeOpt,
            f"set_{option}",
            lambda self, value, name=option: self.options.__setitem__(name, value),
        )

    fake_nlopt.opt = FakeOpt
    monkeypatch.setitem(sys.modules, "nlopt", fake_nlopt)

    backend = importlib.import_module("diffractix.solver.backends.nlopt")
    monkeypatch.setattr(backend, "nlopt", fake_nlopt)
    return backend, fake_nlopt, optimizers


@pytest.mark.parametrize(
    ("backend", "function_name", "method", "options"),
    [
        (Backend.SCIPY, "solve_scipy", "SLSQP", {"maxeval": 100}),
        (Backend.IPOPT, "solve_ipopt", None, {"maxeval": 100}),
        (Backend.NLOPT, "solve_nlopt", "LD_SLSQP", {"maxeval": 100}),
        pytest.param(None, "solve_scipy", None, None, id="default-scipy"),
    ],
)
def test_solver_dispatch_matrix_copies_options(
    monkeypatch,
    backend,
    function_name,
    method,
    options,
):
    from diffractix.solver import solver as solver_module

    solver = object.__new__(Solver)
    solver.objectives = ()
    solver.constraints = ()
    solver.simulation = SimpleNamespace(
        initial_values=np.array([1.0]),
        run=lambda theta: None,
        requirements=(),
        parameter_info={},
        compile_context={},
    )
    solver._compile_objectives = lambda: (Objective(lambda context: 0.0),)
    solver._compile_constraints = lambda constraints: ()
    solver._parameter_bounds = lambda: (np.array([0.0]), np.array([2.0]))
    solver._constraint_bounds = lambda constraints: (np.array([]), np.array([]))
    received = {}

    def solve(problem, method=None, options=None):
        received.update(problem=problem, method=method, options=options)
        numerical_x = (np.array([1.5]) - problem.scaling.x_center) / problem.scaling.x_scale
        return OptimizationResult(
            x=numerical_x,
            success=True,
            cost=0.25,
            message="solved",
            scaling=problem.scaling,
        )

    monkeypatch.setattr(solver_module, function_name, solve)

    solution = (
        solver.solve() if backend is None
        else solver.solve(backend, method=method, options=options)
    )
    assert isinstance(solution, Solution)
    np.testing.assert_array_equal(solution.x, [1.5])
    assert solution.success is True
    assert solution.cost == 0.25
    assert solution.message == "solved"
    assert isinstance(received["problem"], Problem)
    assert received["method"] == method
    assert received["options"] == ({} if options is None else options)
    assert received["options"] is not options
    if options is not None:
        assert options == {"maxeval": 100}


@pytest.mark.parametrize(
    ("entry_point", "backend_module", "function_name", "method"),
    [
        ("solve_scipy", "scipy", "solve_scipy", "SLSQP"),
        ("solve_ipopt", "ipopt", "solve_ipopt", None),
        ("solve_nlopt", "nlopt", "solve_nlopt", "LD_SLSQP"),
    ],
)
def test_lazy_backend_entry_point_matrix(
    monkeypatch,
    entry_point,
    backend_module,
    function_name,
    method,
):
    import sys
    from types import ModuleType

    from diffractix.solver import backends

    received = {}

    def solve(problem, method=None, options=None):
        received.update(problem=problem, method=method, options=options)
        return "result"

    module_name = f"diffractix.solver.backends.{backend_module}"
    module = ModuleType(module_name)
    setattr(module, function_name, solve)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(backends, "_require_package", lambda *args: None)
    options = {"value": 1}

    assert getattr(backends, entry_point)("problem", method, options) == "result"
    assert received == {"problem": "problem", "method": method, "options": options}
    assert received["options"] is not options


def test_lazy_backend_reports_missing_optional_dependency(monkeypatch):
    from diffractix.solver import backends

    def missing(package):
        raise ImportError(package)

    monkeypatch.setattr(backends, "import_module", missing)

    with pytest.raises(ImportError, match=r"diffractix\[nlopt\]"):
        backends.solve_nlopt("problem")


def fake_result(**updates):
    values = {
        "x": np.array([0.5]),
        "success": True,
        "fun": 0.25,
        "message": "done",
        "nit": 3,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_scipy_trust_constr_constructs_native_call(monkeypatch):
    from diffractix.solver.backends import scipy as backend

    problem = make_constrained_problem()
    captured = {"nonlinear_constraints": []}

    def nonlinear_constraint(function, lower, upper, **kwargs):
        captured["nonlinear_constraints"].append(
            (function, lower, upper, kwargs)
        )
        return f"constraint-{len(captured['nonlinear_constraints'])}"

    def minimize(function, x0, **kwargs):
        captured["minimize"] = (function, x0, kwargs)
        return fake_result(x=np.array([1.0, 2.0]), fun=13.0)

    monkeypatch.setattr(backend, "NonlinearConstraint", nonlinear_constraint)
    monkeypatch.setattr(backend, "minimize", minimize)
    options = {"maxiter": 500, "gtol": 1e-8}

    backend.solve_scipy(problem, options=options)

    function, x0, kwargs = captured["minimize"]
    assert function is problem.objective
    np.testing.assert_array_equal(x0, problem.x0)
    assert kwargs["method"] == "trust-constr"
    assert kwargs["jac"] is problem.gradient
    assert kwargs["hess"] is problem.objective_hessian
    np.testing.assert_array_equal(kwargs["bounds"].lb, problem.x_lower)
    np.testing.assert_array_equal(kwargs["bounds"].ub, problem.x_upper)
    assert kwargs["constraints"] == ("constraint-1", "constraint-2")
    assert kwargs["options"] == options
    assert kwargs["options"] is not options

    equality_function, lower, upper, constraint_kwargs = captured[
        "nonlinear_constraints"
    ][0]
    np.testing.assert_array_equal(lower, [3.0])
    np.testing.assert_array_equal(upper, [3.0])
    np.testing.assert_array_equal(equality_function(problem.x0), [-1.0])
    np.testing.assert_array_equal(constraint_kwargs["jac"](problem.x0), [[1.0, -1.0]])
    np.testing.assert_allclose(
        constraint_kwargs["hess"](problem.x0, np.array([7.0])),
        [[0.0, 0.0], [0.0, 0.0]],
    )

    inequality_function, lower, upper, constraint_kwargs = captured[
        "nonlinear_constraints"
    ][1]
    np.testing.assert_array_equal(lower, [2.0, -np.inf, -4.0])
    np.testing.assert_array_equal(upper, [5.0, 10.0, np.inf])
    np.testing.assert_array_equal(inequality_function(problem.x0), [3.0, 1.0, 4.0])
    np.testing.assert_array_equal(
        constraint_kwargs["jac"](problem.x0),
        [[1.0, 1.0], [2.0, 0.0], [0.0, 4.0]],
    )
    np.testing.assert_allclose(
        constraint_kwargs["hess"](problem.x0, np.array([11.0, 13.0, 17.0])),
        [[26.0, 0.0], [0.0, 34.0]],
    )
    assert options == {"maxiter": 500, "gtol": 1e-8}


def test_scipy_subset_hessians_expand_scaled_multipliers_in_original_order(monkeypatch):
    from diffractix.solver.backends import scipy as backend

    # The middle row is a nonlinear equality; rows 0 and 2 are inequalities.
    physical = replace(
        make_scaling_problem(),
        constraint_lower=np.array([0.0, 2.0, 7.0]),
        constraint_upper=np.array([32.0, 2.0, 8.0]),
    )
    problem = physical.scale()
    subsets = []

    def nonlinear_constraint(function, lower, upper, **kwargs):
        subsets.append((function, kwargs))
        return object()

    monkeypatch.setattr(backend, "NonlinearConstraint", nonlinear_constraint)
    monkeypatch.setattr(backend, "minimize", lambda *args, **kwargs: fake_result(x=problem.x0))
    backend.solve_scipy(problem, method="trust-constr")

    equality, inequality = subsets
    y = np.array([0.5, -1.0 / 3.0])  # Physical x = [3, 1].
    np.testing.assert_allclose(equality[0](y), [-1.0 / 6.0])
    np.testing.assert_allclose(inequality[0](y), [0.75, 7.0])
    # Equality v=[6] expands to [0,6,0], then unscales to [0,1/2,0].
    np.testing.assert_allclose(equality[1]["hess"](y, np.array([6.0])), [[0.0, 0.0], [0.0, 9.0]])
    # Inequality v=[8,9] expands to [8,0,9]; only row 0 has curvature.
    np.testing.assert_allclose(inequality[1]["hess"](y, np.array([8.0, 9.0])), [[16.0, 6.0], [6.0, 0.0]])


@pytest.mark.parametrize("equalities", (True, False))
def test_scipy_omits_empty_constraint_subsets(monkeypatch, equalities):
    from diffractix.solver.backends import scipy as backend

    problem = make_constrained_problem()
    lower = np.array([2.0, 3.0, 4.0, 5.0])
    upper = lower.copy() if equalities else lower + 1.0
    problem = replace(problem, constraint_lower=lower, constraint_upper=upper)
    captured = []

    def nonlinear_constraint(*args, **kwargs):
        captured.append((args, kwargs))
        return "constraint"

    monkeypatch.setattr(backend, "NonlinearConstraint", nonlinear_constraint)
    monkeypatch.setattr(
        backend,
        "minimize",
        lambda *args, **kwargs: fake_result(x=np.array([1.0, 2.0])),
    )

    backend.solve_scipy(problem)

    assert len(captured) == 1


def test_scipy_slsqp_uses_first_derivatives_only(monkeypatch):
    from diffractix.solver.backends import scipy as backend

    captured = {}

    def minimize(*args, **kwargs):
        captured.update(kwargs)
        return fake_result()

    monkeypatch.setattr(backend, "minimize", minimize)

    backend.solve_scipy(make_problem(), method="SLSQP")

    assert captured["method"] == "SLSQP"
    assert "hess" not in captured


def test_scipy_unconstrained_problem_registers_no_constraints(monkeypatch):
    from diffractix.solver.backends import scipy as backend

    captured = {}

    def minimize(*args, **kwargs):
        captured.update(kwargs)
        return fake_result()

    monkeypatch.setattr(backend, "minimize", minimize)

    backend.solve_scipy(make_problem())

    assert captured["constraints"] == ()
    assert captured["options"] == {}


def test_scipy_rejects_unknown_method():
    from diffractix.solver.backends import scipy as backend

    with pytest.raises(ValueError, match="Unsupported SciPy method"):
        backend.solve_scipy(make_problem(), method="DOES_NOT_EXIST")


def test_scipy_standardizes_result(monkeypatch):
    from diffractix.solver.backends import scipy as backend

    raw = fake_result(
        x=np.array([0.25]),
        success=False,
        fun=1.5,
        message="stopped",
        nit=9,
    )
    monkeypatch.setattr(backend, "minimize", lambda *args, **kwargs: raw)

    problem = make_problem()
    result = backend.solve_scipy(problem)

    assert isinstance(result, OptimizationResult)
    np.testing.assert_array_equal(result.x, [0.25])
    assert result.success is False
    assert result.cost == 1.5
    assert result.message == "stopped"
    assert result.iterations == 9
    assert result.raw is raw
    assert result.scaling is problem.scaling


def test_ipopt_constructs_problem_and_derivative_callbacks(monkeypatch):
    backend, _, solvers = import_fake_cyipopt_backend(monkeypatch)
    problem = make_constrained_problem()

    backend.solve_ipopt(problem)

    native = solvers[0]
    assert native.kwargs["n"] == problem.n_variables
    assert native.kwargs["m"] == problem.n_constraints
    np.testing.assert_array_equal(native.kwargs["lb"], problem.x_lower)
    np.testing.assert_array_equal(native.kwargs["ub"], problem.x_upper)
    np.testing.assert_array_equal(native.kwargs["cl"], problem.constraint_lower)
    np.testing.assert_array_equal(native.kwargs["cu"], problem.constraint_upper)
    np.testing.assert_array_equal(native.x0, problem.x0)

    callbacks = native.kwargs["problem_obj"]
    x = np.array([1.0, 2.0])
    assert callbacks.objective(x) == problem.objective(x)
    np.testing.assert_array_equal(callbacks.gradient(x), problem.gradient(x))
    np.testing.assert_array_equal(callbacks.constraints(x), problem.constraints(x))
    np.testing.assert_array_equal(
        callbacks.jacobian(x),
        problem.jacobian(x).ravel(),
    )


def test_ipopt_assembles_lower_triangular_lagrangian_hessian(monkeypatch):
    backend, _, solvers = import_fake_cyipopt_backend(monkeypatch)
    problem = make_constrained_problem()
    backend.solve_ipopt(problem)
    callbacks = solvers[0].kwargs["problem_obj"]

    row, col = callbacks.hessianstructure()
    np.testing.assert_array_equal(row, [0, 1, 1])
    np.testing.assert_array_equal(col, [0, 0, 1])
    np.testing.assert_allclose(
        callbacks.hessian(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0, 3.0, 5.0]),
            2.0,
        ),
        [10.0, 0.0, 22.0],
    )


def test_ipopt_forwards_options_without_implicit_hessian_mode(monkeypatch):
    backend, _, solvers = import_fake_cyipopt_backend(monkeypatch)
    options = {"tol": 1e-9, "hessian_approximation": "limited-memory"}

    backend.solve_ipopt(make_problem(), options=options)

    assert solvers[0].options == [
        ("tol", 1e-9),
        ("hessian_approximation", "limited-memory"),
    ]
    assert options == {"tol": 1e-9, "hessian_approximation": "limited-memory"}

    _, _, second_solvers = import_fake_cyipopt_backend(monkeypatch)
    backend.solve_ipopt(make_problem())
    assert second_solvers[0].options == []


def test_ipopt_rejects_method(monkeypatch):
    backend, _, _ = import_fake_cyipopt_backend(monkeypatch)

    with pytest.raises(ValueError, match="does not expose alternative"):
        backend.solve_ipopt(make_problem(), method="anything")


@pytest.mark.parametrize(("status", "success"), [(0, True), (1, True), (-1, False)])
def test_ipopt_standardizes_status_and_result(monkeypatch, status, success):
    backend, _, solvers = import_fake_cyipopt_backend(monkeypatch)
    problem = make_problem()

    # The fake solver is created inside solve_ipopt, so customize it at construction.
    original_problem = backend.cyipopt.Problem

    class ResultProblem(original_problem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.solution = np.array([0.75])
            self.info = {
                "status": status,
                "obj_val": 0.125,
                "status_msg": b"finished",
            }

    backend.cyipopt.Problem = ResultProblem

    result = backend.solve_ipopt(problem)

    assert isinstance(result, OptimizationResult)
    np.testing.assert_array_equal(result.x, [0.75])
    assert result.success is success
    assert result.cost == 0.125
    assert result.message == "finished"
    assert result.raw == {
        "status": status,
        "obj_val": 0.125,
        "status_msg": b"finished",
    }
    assert result.raw is solvers[0].info
    assert result.scaling is problem.scaling


def test_nlopt_default_method_dimensions_and_bounds(monkeypatch):
    backend, nlopt, optimizers = import_fake_nlopt_backend(monkeypatch)
    problem = make_problem()

    backend.solve_nlopt(problem)

    optimizer = optimizers[0]
    assert optimizer.algorithm == nlopt.LD_SLSQP
    assert optimizer.n_variables == problem.n_variables
    np.testing.assert_array_equal(optimizer.lower_bounds, problem.x_lower)
    np.testing.assert_array_equal(optimizer.upper_bounds, problem.x_upper)
    np.testing.assert_array_equal(optimizer.x0, problem.x0)


@pytest.mark.parametrize(
    ("method", "expected"),
    [("LN_COBYLA", 41), ("NLOPT_LD_SLSQP", 40)],
)
def test_nlopt_selects_explicit_method_and_alias(monkeypatch, method, expected):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)

    backend.solve_nlopt(make_problem(), method=method)

    assert optimizers[0].algorithm == expected


def test_nlopt_rejects_unknown_method(monkeypatch):
    backend, _, _ = import_fake_nlopt_backend(monkeypatch)

    with pytest.raises(ValueError, match="Unknown NLopt method"):
        backend.solve_nlopt(make_problem(), method="DOES_NOT_EXIST")


def test_nlopt_objective_callback_populates_gradient(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)
    problem = make_problem()
    backend.solve_nlopt(problem)
    callback = optimizers[0].objective
    gradient = np.zeros(problem.n_variables)

    value = callback(np.array([1.5]), gradient)

    assert value == 2.25
    np.testing.assert_allclose(gradient, [3.0])


def test_nlopt_derivative_free_objective_does_not_evaluate_gradient(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)

    def forbidden_gradient(x):
        raise AssertionError("gradient should not be evaluated")

    problem = make_problem(gradient=forbidden_gradient)
    backend.solve_nlopt(problem, method="LN_COBYLA")

    assert optimizers[0].objective(np.array([1.5]), np.array([])) == 2.25


def test_nlopt_translates_two_sided_and_equality_constraints(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)
    problem = make_constrained_problem()
    backend.solve_nlopt(problem)
    optimizer = optimizers[0]
    x = np.array([1.0, 2.0])

    inequality_callback, _ = optimizer.inequality
    inequality_values = np.empty(4)
    inequality_jacobian = np.empty((4, 2))
    inequality_callback(inequality_values, x, inequality_jacobian)

    np.testing.assert_allclose(inequality_values, [-2.0, -9.0, -1.0, -8.0])
    np.testing.assert_allclose(
        inequality_jacobian,
        [
            [1.0, 1.0],
            [2.0, 0.0],
            [-1.0, -1.0],
            [0.0, -4.0],
        ],
    )

    equality_callback, _ = optimizer.equality
    equality_values = np.empty(1)
    equality_jacobian = np.empty((1, 2))
    equality_callback(equality_values, x, equality_jacobian)
    np.testing.assert_allclose(equality_values, [-4.0])
    np.testing.assert_allclose(equality_jacobian, [[1.0, -1.0]])


def test_nlopt_derivative_free_constraints_do_not_evaluate_jacobian(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)
    base = make_constrained_problem()

    def forbidden_jacobian(x):
        raise AssertionError("Jacobian should not be evaluated")

    problem = replace(base, jacobian=forbidden_jacobian)
    backend.solve_nlopt(problem, method="LN_COBYLA")
    optimizer = optimizers[0]
    x = np.array([1.0, 2.0])
    inequality_values = np.empty(4)
    equality_values = np.empty(1)

    optimizer.inequality[0](inequality_values, x, np.array([]))
    optimizer.equality[0](equality_values, x, np.array([]))

    np.testing.assert_allclose(inequality_values, [-2.0, -9.0, -1.0, -8.0])
    np.testing.assert_allclose(equality_values, [-4.0])


@pytest.mark.parametrize("tolerance", [1e-8, 1e-6])
def test_nlopt_expands_constraint_tolerance_without_mutating_options(
    monkeypatch,
    tolerance,
):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)
    options = {} if tolerance == 1e-8 else {"constraint_tol": tolerance}

    backend.solve_nlopt(make_constrained_problem(), options=options)

    optimizer = optimizers[0]
    np.testing.assert_allclose(optimizer.inequality[1], np.full(4, tolerance))
    np.testing.assert_allclose(optimizer.equality[1], np.full(1, tolerance))
    assert "constraint_tol" not in optimizer.options
    assert options == ({} if tolerance == 1e-8 else {"constraint_tol": tolerance})


def test_nlopt_routes_standard_and_algorithm_specific_options(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)
    options = {"xtol_rel": 1e-8, "maxeval": 500, "inner_maxeval": 10}

    backend.solve_nlopt(make_problem(), options=options)

    assert optimizers[0].options == {"xtol_rel": 1e-8, "maxeval": 500}
    assert optimizers[0].params == {"inner_maxeval": 10}
    assert options == {"xtol_rel": 1e-8, "maxeval": 500, "inner_maxeval": 10}


def test_nlopt_rejects_unknown_option(monkeypatch):
    backend, _, _ = import_fake_nlopt_backend(monkeypatch)

    with pytest.raises(ValueError, match="Unknown NLopt option"):
        backend.solve_nlopt(
            make_problem(),
            options={"definitely_not_an_option": 123},
        )


def test_nlopt_unconstrained_problem_registers_no_constraints(monkeypatch):
    backend, _, optimizers = import_fake_nlopt_backend(monkeypatch)

    backend.solve_nlopt(make_problem())

    assert optimizers[0].inequality is None
    assert optimizers[0].equality is None


@pytest.mark.parametrize(
    ("status_name", "success", "message"),
    [
        ("SUCCESS", True, "Success."),
        ("STOPVAL_REACHED", True, "Stop value reached."),
        ("FTOL_REACHED", True, "Function tolerance reached."),
        ("XTOL_REACHED", True, "Parameter tolerance reached."),
        ("MAXEVAL_REACHED", False, "Maximum evaluations reached."),
        ("MAXTIME_REACHED", False, "Maximum time reached."),
    ],
)
def test_nlopt_status_policy_and_standardized_result(
    monkeypatch,
    status_name,
    success,
    message,
):
    backend, nlopt, _ = import_fake_nlopt_backend(monkeypatch)
    nlopt.next_status = getattr(nlopt, status_name)

    problem = make_problem()
    result = backend.solve_nlopt(problem)

    assert isinstance(result, OptimizationResult)
    np.testing.assert_array_equal(result.x, [0.5])
    assert result.cost == 0.25
    assert result.success is success
    assert result.message == message
    assert result.raw == {
        "status": getattr(nlopt, status_name),
        "algorithm": "fake algorithm",
        "evaluations": 7,
    }
    assert result.scaling is problem.scaling
