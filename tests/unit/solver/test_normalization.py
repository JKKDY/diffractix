"""Objective shorthand and constraint normalization."""

import math
import numpy as np
import pytest
from diffractix.graph import Literal, Parameter
from diffractix.solver.objective import Objective
from diffractix.solver.constraint import Constraint, normalize_to_constraint
from diffractix import Ray, Solver, System

def make_solver():
    return Solver(System(optical_input=Ray(x=Parameter(1).variable(), theta=0, n=1)))


@pytest.mark.parametrize("endpoint", ["lower_bound", "upper_bound"])
def test_constraint_rejects_nan_endpoint(endpoint):
    with pytest.raises(ValueError, match=rf"{endpoint}.*NaN"):
        Constraint(Parameter(1.0), **{endpoint: np.nan})


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf, -0.1])
def test_objective_rejects_nonfinite_or_negative_weight(weight):
    with pytest.raises(ValueError, match="weight.*finite.*>= 0"):
        Objective(Parameter(1.0), weight=weight)


def test_constraint_infinite_endpoints_and_finite_objective_weights_remain_valid():
    parameter = Parameter(1.0)
    constraint = Constraint(parameter, lower_bound=-np.inf, upper_bound=np.inf)
    assert constraint.lower_bound == -np.inf
    assert constraint.upper_bound == np.inf
    for weight in (0.0, np.float32(2.5)):
        assert Objective(parameter, weight=weight).weight == weight


@pytest.mark.parametrize("args, kwargs, error, message", [
    ((), {"value": lambda sim: 1.0}, ValueError, "target must be specified"),
    ((), {"value": 1.0, "target": 0}, TypeError, "value must be a graph expression or a callable"),
    ((), {"value": lambda sim: 1.0, "target": 0, "scale": 0}, ValueError, "scale must be nonzero"),
    ((lambda sim: 1.0,), {"value": lambda sim: 1.0, "target": 0}, ValueError, "positional objectives"),
    ((lambda sim: 1.0,), {"target": 0}, ValueError, "require value"),
    ((lambda sim: 1.0,), {"scale": 2}, ValueError, "require value"),
    ((), {"label": "ambiguous"}, ValueError, "single residual"),
    ((lambda sim: 1.0, lambda sim: 2.0), {"label": "ambiguous"}, ValueError, "single residual"),
    ((Objective(lambda sim: 1.0),), {"label": "override"}, ValueError, "single residual"),
])
def test_target_rejects_ambiguous_or_invalid_shorthand(args, kwargs, error, message):
    solver = make_solver()
    with pytest.raises(error, match=message):
        solver.target(*args, **kwargs)
    assert solver._objectives == []


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



@pytest.mark.parametrize("factory", [
    lambda p: p,
    lambda p: Objective(p, weight=2, label="explicit"),
    lambda p: lambda: p,
    lambda p: lambda result: result.final.x,
])
def test_objective_supported_forms(factory):
    from diffractix.solver.objective import normalize_to_objective
    p = Parameter(1)
    entry = factory(p)
    objective = normalize_to_objective(entry)
    assert isinstance(objective, Objective)
    if isinstance(entry, Objective):
        assert objective is entry
    if not callable(objective.evaluate):
        assert objective.evaluate is p


@pytest.mark.parametrize("entry", [1, None, object(), lambda: 1])
def test_objective_invalid_entries(entry):
    from diffractix.solver.objective import normalize_to_objective
    with pytest.raises(TypeError):
        normalize_to_objective(entry)


@pytest.mark.parametrize("entry", [1, None, object(), Parameter(1), lambda result: result])
def test_constraint_invalid_bare_entries(entry):
    with pytest.raises(TypeError):
        normalize_to_constraint(entry)


def test_zero_argument_constraint_forms_and_reversed_bounds():
    p = Parameter(1)
    comparison = normalize_to_constraint(lambda: p >= 0)
    explicit = normalize_to_constraint(Constraint(lambda: p, lower_bound=0, label="named"))
    assert comparison.evaluate is explicit.evaluate is p
    assert explicit.label == "named"
    assert comparison.lower_bound == explicit.lower_bound == 0
    with pytest.raises(ValueError, match="lower_bound"):
        Constraint(p, lower_bound=2, upper_bound=1)
    with pytest.raises(TypeError, match="Node"):
        normalize_to_constraint(Constraint(lambda: 1))


@pytest.mark.parametrize("function,expected", [
    (lambda: 1, 0), (lambda value: value, 1),
    (lambda *, required: required, None),
    (lambda value=1: value, None), (lambda *args: args, None),
    (lambda first, second: first + second, None),
])
def test_callable_arity_requires_one_unambiguous_call_form(function, expected):
    from diffractix.solver.utils import callable_arity
    if expected is None:
        with pytest.raises(TypeError, match="exactly zero or one"):
            callable_arity(function)
    else:
        assert callable_arity(function) == expected


def test_callable_objects_and_bound_methods_normalize_like_functions():
    from diffractix.solver.objective import normalize_to_objective
    from diffractix.solver.utils import callable_arity

    class Residual:
        def __call__(self, result):
            return result

        def constant(self):
            return Parameter(2)

    entry = Residual()
    assert callable_arity(entry) == 1
    assert callable_arity(entry.constant) == 0
    assert normalize_to_objective(entry).evaluate is entry
    assert normalize_to_objective(entry.constant).evaluate.value == 2



def test_value_matching_and_residual_shorthand_match_explicit_objective_derivatives():
    from diffractix.backend import xp
    from tests.helpers.backend import capture_problem
    p = Parameter(0.5).variable()
    simulation = System(optical_input=Ray(x=p, theta=0, n=1)).build()
    solvers = [Solver(simulation) for _ in range(4)]
    solvers[0].target(value=lambda result: result.final.x ** 2, target=0, scale=2, label="match")
    solvers[1].target(lambda result: result.final.x ** 2 / 2, label="match")
    solvers[2].target(Objective(lambda result: result.final.x ** 2 / 2, label="match"))
    solvers[3].target(value=p ** 2, target=0, scale=2, label="match")   # a graph expression of the Parameter
    for solver in solvers:
        problem = capture_problem(solver)
        theta = xp.array([2.])
        assert problem.objective(theta) == pytest.approx(4)
        np.testing.assert_allclose(problem.gradient(theta), [8])
        np.testing.assert_allclose(problem.objective_hessian(theta), [[12]])
        assert solver._cache.objective_descriptors[0].label == "match"


def test_require_validation_is_atomic_and_chaining_preserves_entries():
    solver = make_solver()
    p = Parameter(1)
    with pytest.raises(TypeError):
        solver.require(p >= 0, object())
    assert solver._constraints == []
    assert solver.require(p >= 0) is solver
    assert len(solver._constraints) == 1


@pytest.mark.parametrize("constructor", [Objective, Constraint])
@pytest.mark.parametrize("invalid", [1, None, object()])
def test_entry_constructor_requires_graph_or_callable(constructor, invalid):
    with pytest.raises(TypeError, match="Node or callable"):
        constructor(invalid)



@pytest.mark.parametrize("scale", [np.array([1., 0.]), np.array([[0.], [2.]]), np.array(0.)])
def test_value_matching_rejects_any_zero_scale_entry_without_adding_objective(scale):
    solver = make_solver()
    with pytest.raises(ValueError, match="scale must be nonzero"):
        solver.target(value=lambda sim: sim.final.x, target=np.array([1., 2.]), scale=scale)
    assert solver._objectives == []


@pytest.mark.parametrize("use_jit", [False, True])
def test_array_matching_broadcasts_and_preserves_analytical_derivatives(use_jit):
    from diffractix.backend import xp
    parameter = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))
    target, scale = np.array([1., 3.]), np.array([1., 2.])
    solver.target(value=lambda sim: sim.final.x, target=target, scale=scale)
    callbacks = solver._build_cache(use_jit, solver.simulation.initial_values.values).callbacks
    point = xp.array([2.])
    # Cost = (x-1)^2 + (x-3)^2/4.
    assert callbacks.objective(point) == pytest.approx(1.25)
    np.testing.assert_allclose(callbacks.gradient(point), [1.5])
    np.testing.assert_allclose(callbacks.objective_hessian(point), [[2.5]])
    np.testing.assert_array_equal(target, [1., 3.])
    np.testing.assert_array_equal(scale, [1., 2.])


def objective_at(solver, theta):
    from diffractix.backend import xp
    from tests.helpers.backend import capture_problem
    return float(capture_problem(solver).objective(xp.array([theta])))


@pytest.mark.parametrize("make_value", [
    pytest.param(lambda p: p, id="expression"),
    pytest.param(lambda p: (lambda sim: sim.final.x), id="callable"),
])
@pytest.mark.parametrize("target, at, residual", [
    (2.0, 3.0, 0.5),    # relative to |target|
    (-4.0, -2.0, 0.5),  # the sign of the target does not matter
    (0.0, 3.0, 3.0),    # a zero target keeps unit scale
])
def test_value_matching_scale_defaults_to_the_target_magnitude_or_one(make_value, target, at, residual):
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.target(value=make_value(p), target=target)
    assert objective_at(solver, at) == pytest.approx(residual ** 2)


def test_default_scale_is_elementwise_for_array_targets_and_explicit_scale_wins():
    p = Parameter(0.5).variable()
    solver = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    solver.target(value=p * np.array([1., 1.]), target=np.array([2., 0.]))
    assert objective_at(solver, 3.0) == pytest.approx(0.5 ** 2 + 3.0 ** 2)
    explicit = Solver(System(optical_input=Ray(x=p, theta=0, n=1)))
    explicit.target(value=p, target=2.0, scale=4.0)
    assert objective_at(explicit, 3.0) == pytest.approx((1 / 4) ** 2)


def test_require_label_names_a_single_comparison():
    solver, p = make_solver(), Parameter(1)
    assert solver.require(p >= 0, label="positive") is solver
    assert [constraint.label for constraint in solver._constraints] == ["positive"]
    assert solver._constraints[0].lower_bound == 0


@pytest.mark.parametrize("make_arguments", [
    pytest.param(lambda p: (), id="none"),
    pytest.param(lambda p: (p >= 0, p <= 1), id="two"),
    pytest.param(lambda p: (Constraint(p, lower_bound=0),), id="constraint"),
])
def test_require_label_rejects_anything_but_one_comparison_without_adding(make_arguments):
    solver, p = make_solver(), Parameter(1)
    with pytest.raises(ValueError, match="label requires a single comparison"):
        solver.require(*make_arguments(p), label="name")
    assert solver._constraints == []


def test_require_label_with_a_raw_node_fails_without_adding():
    solver = make_solver()
    with pytest.raises(TypeError):
        solver.require(Parameter(1), label="name")
    assert solver._constraints == []
