"""Shared numerical cases executed unchanged in each selected backend process."""
import numpy as np


def _encode(value):
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    value = np.asarray(value)
    if np.iscomplexobj(value):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()}
    return value.tolist()


def _integration():
    """Small current-API cases independent of historical examples and fixtures."""
    from diffractix import GaussianBeam, Parameter, Ray, Solver, Space, System
    from diffractix.backend import grad, hessian, jacobian, jit, xp
    from diffractix.graph import Symbol, compile_graph

    p, q = Parameter(0.3).variable(), Parameter(-0.2).variable()
    program = compile_graph((p**3 + q**2, p*q + Symbol("gain")))
    indices = program.parameter_indices
    def graph(theta):
        return xp.stack(program.evaluate(theta, {"gain": 0.7}))
    theta = np.zeros(2)
    theta[indices[id(p)]], theta[indices[id(q)]] = 0.3, -0.2
    theta = xp.array(theta)
    scalar = lambda value: xp.sum(graph(value)**2)
    graph_values = {
        "values": graph(theta), "jacobian": jacobian(graph)(theta),
        "gradient": grad(scalar)(theta), "hessian": hessian(scalar)(theta),
        "jit": jit(graph)(theta),
        "p_index": indices[id(p)], "q_index": indices[id(q)],
    }

    distance = Parameter(0.03).variable(bounds=(0.005, 0.1))
    simulation = System(Space(d=distance), optical_input=GaussianBeam.from_waist(0.0005, 1.064e-6)).build()
    def optics(value):
        state = simulation.run(value).final
        return xp.stack((state.w, xp.real(state.q), xp.imag(state.q)))
    point = xp.array([0.03])
    optical_values = {"values": optics(point), "jacobian": jacobian(optics)(point),
                      "hessian": hessian(lambda value: xp.sum(optics(value)**2))(point),
                      "jit": jit(optics)(point)}
    position = Parameter(0.001).variable(bounds=(-0.005, 0.005))
    ray = Ray(x=position, theta=0.01)
    solver = Solver(System(Space(d=0.1), optical_input=ray))
    solver.target(lambda result: (result.final.x - 0.003) / 0.001)
    solution = solver.solve(use_jit=True, options={"ftol": 1e-12})
    return {"graph": graph_values, "optics": optical_values,
            "solve": {"x": solution.x, "cost": solution.cost, "success": solution.success,
                      "feasible": solution.feasible, "final_x": solution.run().final.x}}


def evaluate_cases(group):
    from diffractix.backend import current_backend
    if group != "integration":
        raise ValueError(f"Unknown numerical case group: {group!r}")
    values = _integration()
    return {"backend": current_backend(), "values": _encode(values)}
