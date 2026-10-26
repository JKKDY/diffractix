"""Ray propagation through real optical elements, composites, and compiled graphs."""

import numpy as np
import pytest

from diffractix import Interface, Parameter, Ray, Slab, Space, System, ThickLens, ThinLens
from diffractix.backend import grad, hessian, jacobian, jit, xp


@pytest.fixture
def source():
    return Ray(x=1.2e-3, theta=0.018, n=1.33).numeric()


def assert_trace(result, *, x, theta, n, z):
    assert all(isinstance(state, Ray) for state in result.states)
    np.testing.assert_allclose(result.x, x, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.theta, theta, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.n, n, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.z, z, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("focal_length", [0.12, -0.12])
def test_ray_space_and_thin_lens_follow_physical_angle_convention(source, focal_length):
    first, lens, last = Space(d=0.025), ThinLens(f=focal_length), Space(d=0.015)
    simulation = System(first, lens, last, input=source, ambient_n=source.n).build()
    result = simulation.run()
    lens_height = source.x + 0.025*source.theta
    output_angle = source.theta - lens_height/focal_length

    assert_trace(
        result,
        x=[source.x, lens_height, lens_height, lens_height+0.015*output_angle],
        theta=[source.theta, source.theta, output_angle, output_angle],
        n=[source.n]*4,
        z=[0.0, 0.025, 0.025, 0.04],
    )
    assert result.at(lens) is result.states[1]
    assert result.after(lens) is result.states[2]
    np.testing.assert_array_equal(source.vector, [1.2e-3, 0.018])
    assert source.n == 1.33


@pytest.mark.parametrize("n1, n2", [(1.0, 1.5), (1.5, 1.0)])
def test_ray_planar_interface_preserves_paraxial_snell_invariant(n1, n2):
    source = Ray(x=1.2e-3, theta=-0.018, n=n1).numeric()
    interface = Interface(n1=n1, n2=n2)
    result = System(interface, input=source, ambient_n=n1).build().run()

    assert_trace(
        result, x=[source.x, source.x], theta=[source.theta, n1*source.theta/n2],
        n=[n1, n2], z=[0.0, 0.0],
    )
    assert result.at(interface).n == pytest.approx(n1)
    assert result.after(interface).n == pytest.approx(n2)
    assert n2*result.final.theta == pytest.approx(n1*source.theta)


def test_ray_planar_interface_variable_index_derivatives_through_simulation():
    source = Ray(x=1.2e-3, theta=0.018, n=1.0).numeric()
    index = Parameter(1.5).variable(bounds=(1.1, 2.0))
    distance = 0.02
    simulation = System(Interface(n1=1.0, n2=index), Space(d=distance), input=source).build()
    evaluate = lambda theta: simulation.run(theta).final.vector
    angle = lambda theta: simulation.run(theta).final.theta
    for value in (1.2, 1.8):
        theta = xp.array([value])
        expected_angle = source.theta / value
        for function in (evaluate, jit(evaluate)):
            np.testing.assert_allclose(function(theta), [source.x + distance * expected_angle, expected_angle])
        for function in (jacobian(evaluate), jit(jacobian(evaluate))):
            np.testing.assert_allclose(
                function(theta), [[-distance * source.theta / value**2], [-source.theta / value**2]],
                rtol=1e-12, atol=1e-15,
            )
        for function in (hessian(angle), jit(hessian(angle))):
            np.testing.assert_allclose(function(theta), [[2 * source.theta / value**3]], rtol=1e-12)


@pytest.mark.scipy
def test_solver_can_optimize_planar_interface_refractive_index():
    from diffractix import Solver

    source = Ray(x=1e-3, theta=0.018, n=1.0).numeric()
    index = Parameter(1.3).variable(bounds=(1.1, 2.0))
    interface = Interface(n1=1.0, n2=index)
    simulation = System(interface, input=source).build()
    solver = Solver(simulation).target(lambda ctx: ctx.after(interface).theta / source.theta - 1 / 1.6)
    solution = solver.solve(method="SLSQP", use_jit=True, options={"ftol": 1e-12})

    assert solution.success, solution.message
    assert solution.feasible
    assert solution[index] == pytest.approx(1.6, abs=1e-6)
    np.testing.assert_allclose(solution.run().final.theta, source.theta / 1.6, rtol=1e-6)
    assert index.value == 1.3


@pytest.mark.parametrize("radius", [0.08, -0.08])
def test_ray_curved_interface_uses_height_and_surface_curvature(source, radius):
    n2 = 1.52
    interface = Interface(n1=source.n, n2=n2, R=radius)
    result = System(interface, input=source, ambient_n=source.n).build().run()
    expected_angle = (source.n*source.theta - (n2-source.n)*source.x/radius)/n2

    assert_trace(
        result, x=[source.x, source.x], theta=[source.theta, expected_angle],
        n=[source.n, n2], z=[0.0, 0.0],
    )


def test_ray_slab_tracks_internal_medium_and_restores_exit_angle(source):
    thickness, material_index = 0.008, 1.52
    slab = Slab(d=thickness, n=material_index)
    result = System(slab, input=source, ambient_n=source.n).build().run()
    inside_angle = source.n*source.theta/material_index
    exit_height = source.x + thickness*inside_angle

    assert_trace(
        result,
        x=[source.x, source.x, exit_height, exit_height],
        theta=[source.theta, inside_angle, inside_angle, source.theta],
        n=[source.n, material_index, material_index, source.n],
        z=[0.0, 0.0, thickness, thickness],
    )
    # Composite traces are addressed by their resolved child elements.
    assert result.at(slab.front) is result.initial
    assert result.after(slab.back) is result.final
    assert result.at(slab.body).n == pytest.approx(material_index)


def test_ray_thick_lens_matches_two_refractions_and_internal_translation(source):
    thickness, material_index, first_radius, last_radius = 0.006, 1.52, 0.08, -0.11
    lens = ThickLens(d=thickness, n=material_index, R1=first_radius, R2=last_radius)
    result = System(lens, input=source, ambient_n=source.n).build().run()
    inside_angle = (
        source.n*source.theta - (material_index-source.n)*source.x/first_radius
    )/material_index
    exit_height = source.x + thickness*inside_angle
    exit_angle = (
        material_index*inside_angle - (source.n-material_index)*exit_height/last_radius
    )/source.n

    assert_trace(
        result,
        x=[source.x, source.x, exit_height, exit_height],
        theta=[source.theta, inside_angle, inside_angle, exit_angle],
        n=[source.n, material_index, material_index, source.n],
        z=[0.0, 0.0, thickness, thickness],
    )
    assert result.after(lens.front) is result.states[1]
    assert result.after(lens.body) is result.states[2]
    assert result.after(lens.back) is result.final


def test_ray_variable_index_updates_trace_and_reuses_jitted_simulation():
    source = Ray(x=1.2e-3, theta=0.018).numeric({"ambient_n": 1.0})
    index = Parameter(1.5).variable(bounds=(1.1, 1.8))
    front, body, back = Interface(n1=1.0, n2=index), Space(d=0.01, n=index), Interface(n1=index, n2=1.0)
    simulation = System(front, body, back, input=source).build()
    compiled = jit(lambda theta: simulation.run(theta).final.vector)

    for medium in (1.2, 1.7):
        values = xp.array([medium])
        result = simulation.run(values)
        inside_angle = source.theta/medium
        exit_height = source.x + 0.01*inside_angle
        assert_trace(
            result,
            x=[source.x, source.x, exit_height, exit_height],
            theta=[source.theta, inside_angle, inside_angle, source.theta],
            n=[1.0, medium, medium, 1.0], z=[0.0, 0.0, 0.01, 0.01],
        )
        np.testing.assert_allclose(compiled(values), [exit_height, source.theta], atol=1e-15)
    assert index.value == 1.5
    assert source.n == 1.0


def test_ray_simulation_parameter_derivatives_and_jit(source):
    distance = Parameter(0.025).variable(bounds=(0.01, 0.06))
    focal_length = Parameter(0.12).variable(bounds=(0.08, 0.20))
    thickness, material_index = 0.008, 1.52
    simulation = System(
        Space(d=distance), ThinLens(f=focal_length), Slab(d=thickness, n=material_index),
        input=source, ambient_n=source.n,
    ).build()
    assert simulation.graph.variables == (distance, focal_length)

    def trace(values):
        return simulation.run(values).final.vector

    values = xp.array([0.03, 0.15])
    d, f = np.asarray(values)
    height = source.x + d*source.theta
    angle = source.theta - height/f
    # Refraction changes the angle only inside the slab; its translation is physical d*theta.
    effective_distance = thickness*source.n/material_index
    expected_jacobian = np.array([
        [source.theta*(1-effective_distance/f), effective_distance*height/f**2],
        [-source.theta/f, height/f**2],
    ])
    angle_hessian = np.array([
        [0.0, source.theta/f**2], [source.theta/f**2, -2*height/f**3],
    ])
    for function, expected in (
        (trace, [height+effective_distance*angle, angle]),
        (grad(lambda t: trace(t)[0]), expected_jacobian[0]),
        (jacobian(trace), expected_jacobian),
        (hessian(lambda t: trace(t)[0]), effective_distance*angle_hessian),
        (hessian(lambda t: trace(t)[1]), angle_hessian),
    ):
        np.testing.assert_allclose(function(values), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jit(function)(values), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(simulation.initial_values, [0.025, 0.12])
