"""Shared numerical cases executed unchanged in each selected backend process."""
import numpy as np


def _encode(value):
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    value = np.asarray(value)
    if np.iscomplexobj(value):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()}
    return value.tolist()


def _graph_case():
    from diffractix.backend import xp
    from diffractix.graph import InputNode, Literal, Parameter, Symbol, compile_ast

    x, y, offset = Parameter(0.6).variable(), Parameter(-0.2).variable(), Parameter(0.25)
    alias = InputNode(x)
    expression = Symbol(("runtime", "gain")) * (alias**3 + y**2) + Symbol("offset")
    graph = compile_ast((expression, expression.sin(), x+y),
                        context={"offset": offset*2 + Literal(0.0)})
    # Verify frozen compilation and aliases while differentiating live inputs.
    alias.node = Literal(99.0)
    offset.value = 99.0
    x.fixed()

    def values(theta):
        return graph.evaluate(theta[:2], bindings={("runtime", "gain"): theta[2]})

    array_graph = compile_ast((y**2 * Symbol("array") + (1+2j)*y,))

    def array_values(theta):
        return array_graph.evaluate(theta[:1], bindings={
            "array": theta[1:3] + 1j*theta[3:5],
        })

    return graph, values, array_values


def _optical_cases():
    from diffractix import (
        GaussianAperture, GaussianBeam, Interface, Objective, Parameter,
        Ray, RayBundle, Slab, Solver, Space, System, ThickLens, ThinLens,
    )
    from diffractix.backend import xp
    from diffractix.solver import Constraint
    from tests.helpers.backend import capture_problem

    distance = Parameter(0.03).variable(bounds=(0.01, 0.06))
    focal = Parameter(0.12).variable(bounds=(0.08, 0.2))
    source = GaussianBeam(q=0.015+0.15j, wavelength=1.064e-6, n=1.0).numeric()
    target = GaussianBeam.from_waist(0.0004, source.wavelength).numeric()
    space, lens, output = Space(d=distance), ThinLens(f=focal), Space(d=0.04)
    gaussian = System(space, lens, output, input=source).build()
    ray = System(space, lens, output, input=Ray(x=0.001, theta=0.02).numeric({"ambient_n": 1.0})).build()
    bundle = System(space, lens, output, input=RayBundle(
        x=np.array([0.001, -0.002]), theta=np.array([0.02, -0.01]),
    ).numeric({"ambient_n": 1.0})).build()
    refractive = System(
        Interface(n1=1, n2=1.5, R=0.11), Space(d=0.01, n=1.5),
        Interface(n1=1.5, n2=1, R=-0.13), Slab(d=0.007, n=1.4),
        ThickLens(d=0.006, n=1.52, R1=0.14, R2=-0.18),
        space, lens, output, input=source,
    ).build()
    aperture = System(space, GaussianAperture(a=0.001), lens, output, input=source).build()

    def beam_properties(simulation, theta):
        result = simulation.run(theta)
        state = result.final
        return xp.stack((
            xp.real(state.q), xp.imag(state.q), xp.real(state.w), xp.real(state.R),
            state.z_r, state.z0, xp.real(state.w0), state.gouy_phase,
            xp.real(state.NA), state.overlap_with(target), xp.real(result.z[-1]),
        ))

    def phase_space(simulation, theta):
        state = simulation.run(theta).final
        return xp.stack((state.x, state.theta))

    solver = Solver(gaussian)
    solver.target(
        Objective(lambda ctx: ctx.after(output).w*1000-0.4, weight=2.0, label="radius"),
        Objective((distance-0.025)/0.05, weight=0.5, label="distance"),
    )
    solver.require(
        Constraint(lambda ctx: ctx.after(output).w*1000, lower_bound=0.1, upper_bound=2),
        Constraint(lambda ctx: ctx.after(output).z0*10, lower_bound=-0.5, upper_bound=0.5),
    )
    problem = capture_problem(solver)
    assert problem.n_constraints == 2
    return {
        "gaussian": lambda t: beam_properties(gaussian, t),
        "ray": lambda t: phase_space(ray, t),
        "bundle": lambda t: phase_space(bundle, t),
        "refractive": lambda t: beam_properties(refractive, t),
        "aperture": lambda t: beam_properties(aperture, t),
    }, problem


def _derivative_values(function, theta, multipliers):
    from diffractix.backend import grad, hessian, jacobian, jacrev, xp

    scalar = lambda t: xp.sum(function(t)**2)
    weighted = lambda t: xp.dot(multipliers, function(t))
    return {
        "values": function(theta), "gradient": grad(scalar)(theta),
        "jacobian": jacobian(function)(theta), "jacrev": jacrev(function)(theta),
        "hessian": hessian(scalar)(theta), "weighted_hessian": hessian(weighted)(theta),
    }


def _representative():
    from diffractix.backend import current_backend, grad, hessian, jacobian, jit, require, xp

    graph, graph_values, array_values = _graph_case()
    from diffractix import GaussianBeam, Parameter
    from diffractix.graph import compile_ast

    # Declarative graph constants must be Python scalars, not backend arrays.
    beam_scale = float(GaussianBeam(q=0.015+0.15j, wavelength=1.064e-6, n=1.0).z_r)
    captured = compile_ast((Parameter(0.03).variable()/beam_scale,))
    optics, problem = _optical_cases()
    graph_points = xp.array([[0.6, -0.2, 1.3], [0.4, 0.3, 0.9], [-0.2, 0.5, 1.1]])
    optical_points = xp.array([[0.03, 0.12], [0.025, 0.10], [0.04, 0.15]])
    result = {"graph": {}, "optics": {}, "solver": {}}
    result["scalar_capture"] = {
        "values": captured.evaluate(xp.array([0.03])),
        "gradient": grad(lambda t: xp.sum(captured.evaluate(t)))(xp.array([0.03])),
    }
    graph_weights = xp.array([0.7, -0.2, 1.1])
    for index, point in enumerate(graph_points):
        result["graph"][str(index)] = _derivative_values(graph_values, point, graph_weights)
    for name, function in optics.items():
        result["optics"][name] = xp.stack([function(t) for t in optical_points])
    for index, point in enumerate(optical_points):
        result["solver"][str(index)] = {
            "objective": problem.objective(point), "gradient": problem.gradient(point),
            "constraints": problem.constraints(point), "jacobian": problem.jacobian(point),
            "hessian": problem.objective_hessian(point),
            "weighted_hessian": problem.constraint_hessian(point, xp.array([0.7, -1.2])),
        }
    t = xp.array([0.4, 0.2, 0.7, 0.1, -0.3])
    array_cost = lambda t: xp.sum(xp.real(array_values(t)*xp.conj(array_values(t))))
    complex_cost = lambda t: xp.real((t[0]+1j*t[1])**2 * xp.conj((t[0]+1j*t[1])**2))
    scalar = lambda t: xp.sum(graph_values(t)**2)
    result["array"] = {"values": array_values(t), "gradient": grad(array_cost)(t),
                       "hessian": hessian(array_cost)(t)}
    result["complex"] = {"value": complex_cost(graph_points[0, :2]),
                         "gradient": grad(complex_cost)(graph_points[0, :2]),
                         "hessian": hessian(complex_cost)(graph_points[0, :2])}
    fourth_graph = grad(grad(grad(grad(
        lambda x: scalar(xp.stack((x, -0.2+0*x, 1.3+0*x)))))))
    nested = hessian(lambda t: xp.sum(grad(scalar)(t)**2))
    result["higher"] = {
        "fourth_polynomial": grad(grad(grad(grad(lambda x: x**6))))(0.5),
        "fourth_graph": fourth_graph(0.6), "nested": nested(graph_points[0]),
    }
    result["metadata"] = {"variables": len(graph.variables), "symbols": len(graph.symbols)}

    # Transform identical numerical callbacks, including runtime binding and objects.
    callbacks = {
        "graph": (graph_values, graph_points[0]),
        "graph_gradient": (grad(scalar), graph_points[0]),
        "graph_jacobian": (jacobian(graph_values), graph_points[0]),
        "objective": (problem.objective, optical_points[0]),
        "gradient": (problem.gradient, optical_points[0]),
        "constraints": (problem.constraints, optical_points[0]),
        "jacobian": (problem.jacobian, optical_points[0]),
        "hessian": (problem.objective_hessian, optical_points[0]),
        "weighted_hessian": (lambda t: problem.constraint_hessian(t, xp.array([0.7, -1.2])), optical_points[0]),
        "nested": (nested, graph_points[0]),
        "fourth": (fourth_graph, graph_points[0, 0]),
        **{name: (fn, optical_points[0]) for name, fn in optics.items()},
    }
    result["jit"] = {name: {"eager": function(point), "compiled": jit(function)(point)}
                     for name, (function, point) in callbacks.items()}
    function = lambda x, power=2: x**power
    result["jit_identity"] = {"function": jit(function) is function,
                              "decorator": jit(static_argnums=1)(function) is function}
    @jit(static_argnums=1)
    def decorated(theta, power):
        return xp.stack((theta[0]**power, theta[0]*theta[1]))

    from diffractix.backend import jacrev
    decorated_values = lambda t: decorated(t, 3)
    result["decorator"] = {
        "values": decorated_values(xp.array([2.0, 3.0])),
        "jacobian": jacobian(decorated_values)(xp.array([2.0, 3.0])),
        "jacrev": jacrev(decorated_values)(xp.array([2.0, 3.0])),
    }
    checks = {}
    for name, condition in (("python_true", True), ("numerical_true", xp.array(True)),
                            ("python_false", False), ("numerical_false", xp.array(False))):
        try:
            checks[name] = require(condition, "deliberate validation failure")
        except ValueError as error:
            checks[name] = str(error)
    result["require"] = checks
    if current_backend() == "jax":
        import jax

        def guarded(x):
            require(x > 0, "must be positive")
            return x*x

        result["jax"] = {
            "float64": str(xp.array(1.0).dtype),
            "dynamic_invalid": jit(guarded)(-2.0),
            "guarded_gradient": jit(grad(guarded))(2.0),
            "guarded_hessian": jit(hessian(guarded))(2.0),
            "batched_guarded": jit(jax.vmap(guarded))(xp.array([1.0, -2.0, 3.0])),
            "require_jaxpr": str(jax.make_jaxpr(guarded)(2.0)),
            "batched_graph": jit(jax.vmap(graph_values))(graph_points),
            **{f"batched_{name}": jit(jax.vmap(fn))(optical_points)
               for name, fn in optics.items()},
        }
        try:
            jit(lambda x: (require(False, "static failure"), x)[1])(1.0)
        except ValueError as error:
            result["jax"]["static_false"] = str(error)
    return result


def _advanced():
    from diffractix.backend import xp
    from tests.helpers.backend import capture_problem, example_before_solve

    namespace = example_before_solve("multistation_delivery")
    solver = namespace["solver"]
    problem = capture_problem(solver)
    initial = problem.x0
    width = problem.x_upper - problem.x_lower
    perturbation = 0.005*width*np.cos(np.arange(len(initial)))
    points = (initial, np.clip(initial+perturbation, problem.x_lower, problem.x_upper),
              np.clip(initial-perturbation, problem.x_lower, problem.x_upper))
    return {str(index): {
        "radii": solver.simulation.run(xp.array(t)).w,
        "objective": problem.objective(xp.array(t)),
        "constraints": problem.constraints(xp.array(t)),
        "gradient": problem.gradient(xp.array(t)),
        "jacobian": problem.jacobian(xp.array(t)),
    } for index, t in enumerate(points)}


def evaluate_cases(group):
    from diffractix.backend import current_backend

    values = _representative() if group == "representative" else _advanced()
    # Convert only completed eager results, never values inside differentiation.
    metadata = {key: values.pop(key) for key in ("metadata", "jit_identity", "require", "jax")
                if key in values}
    return {"backend": current_backend(), "values": _encode(values), **_encode(metadata)}
