"""Single-ray physical-angle ABCD propagation and backend-neutral derivatives."""

import numpy as np
import pytest

from diffractix import Ray
from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.input import Ray as InputRay, ParaxialState


def test_ray_construction_default_medium_and_public_exports():
    ray = Ray(x=1.2e-3, theta=-0.018).numeric({"ambient_n": 1.0})

    assert InputRay is Ray
    assert isinstance(ray, ParaxialState)
    assert ray.x == 1.2e-3
    assert ray.theta == -0.018
    assert ray.n == 1.0
    np.testing.assert_array_equal(ray.vector, [1.2e-3, -0.018])
    assert str(ray).startswith("Ray(")


@pytest.mark.parametrize("z", [0.0, 0.025, -0.025])
def test_x_at_z_uses_physical_angle_even_in_a_refractive_medium(z):
    ray = Ray(x=1.2e-3, theta=-0.018, n=1.5).numeric()

    assert ray.x_at_z(z) == pytest.approx(1.2e-3 - 0.018*z)
    assert (ray.x, ray.theta, ray.n) == (1.2e-3, -0.018, 1.5)


def test_propagation_returns_an_independent_ray_and_tracks_output_medium():
    ray = Ray(x=1.2e-3, theta=0.018, n=1.0).numeric()
    propagated = ray.propagate(1.0, 0.0, 0.0, 1.0/1.5, n=1.5)

    assert isinstance(propagated, Ray)
    assert propagated is not ray
    np.testing.assert_allclose(propagated.vector, [1.2e-3, 0.012], atol=1e-15)
    assert propagated.n == 1.5
    assert (ray.x, ray.theta, ray.n) == (1.2e-3, 0.018, 1.0)
    propagated.x = 0.0
    assert ray.x == 1.2e-3


def test_free_space_propagation_does_not_divide_distance_by_index():
    ray = Ray(x=1.2e-3, theta=0.018, n=1.5).numeric()
    propagated = ray.propagate(1.0, 0.025, 0.0, 1.0, n=1.5)

    np.testing.assert_allclose(propagated.vector, [1.65e-3, 0.018], atol=1e-15)
    assert propagated.n == 1.5


@pytest.mark.parametrize("focal_length", [0.12, -0.12])
def test_thin_lens_changes_angle_not_height(focal_length):
    ray = Ray(x=1.2e-3, theta=0.018, n=1.33).numeric()
    propagated = ray.propagate(1.0, 0.0, -1.0/focal_length, 1.0, n=ray.n)

    np.testing.assert_allclose(
        propagated.vector, [ray.x, ray.theta - ray.x/focal_length], atol=1e-15,
    )
    assert propagated.n == ray.n


@pytest.mark.parametrize("n1, n2", [(1.0, 1.5), (1.5, 1.0)])
def test_planar_interface_obeys_paraxial_snell_law(n1, n2):
    ray = Ray(x=1.2e-3, theta=-0.018, n=n1).numeric()
    propagated = ray.propagate(1.0, 0.0, 0.0, n1/n2, n=n2)

    np.testing.assert_allclose(propagated.vector, [ray.x, n1*ray.theta/n2], atol=1e-15)
    assert propagated.n == n2
    assert n2*propagated.theta == pytest.approx(n1*ray.theta)


@pytest.mark.parametrize("radius", [0.08, -0.08])
def test_curved_interface_includes_surface_power(radius):
    ray = Ray(x=1.2e-3, theta=0.018, n=1.0).numeric()
    outgoing_index = 1.52
    propagated = ray.propagate(
        1.0, 0.0, (ray.n-outgoing_index)/(radius*outgoing_index),
        ray.n/outgoing_index, n=outgoing_index,
    )
    expected_angle = (ray.n*ray.theta - (outgoing_index-ray.n)*ray.x/radius)/outgoing_index

    np.testing.assert_allclose(propagated.vector, [ray.x, expected_angle], atol=1e-15)
    assert propagated.n == outgoing_index


def test_sequential_propagation_matches_ordered_matrix_product():
    ray = Ray(x=1.2e-3, theta=-0.018).numeric({"ambient_n": 1.0})
    interface = np.array([[1.0, 0.0], [-4.0, 2.0/3.0]])
    space = np.array([[1.0, 0.025], [0.0, 1.0]])

    first = ray.propagate(*interface.ravel(), n=1.5)
    sequential = first.propagate(*space.ravel(), n=1.5)
    combined = ray.propagate(*(space @ interface).ravel(), n=1.5)

    np.testing.assert_allclose(sequential.vector, space @ interface @ ray.vector, atol=1e-15)
    np.testing.assert_allclose(combined.vector, sequential.vector, atol=1e-15)
    assert sequential.n == combined.n == 1.5


def test_on_axis_ray_stays_on_axis_through_a_powered_system():
    ray = Ray(x=0.0, theta=0.0).numeric({"ambient_n": 1.0})
    # A curved air/glass boundary followed by 25 mm of propagation in glass.
    propagated = ray.propagate(0.9, 0.025*2.0/3.0, -4.0, 2.0/3.0, n=1.5)

    np.testing.assert_array_equal(propagated.vector, [0.0, 0.0])
    assert propagated.n == 1.5


@pytest.mark.parametrize("focal_length", [0.12, -0.12])
def test_propagation_derivatives_and_jit_for_source_distance_and_focal_length(focal_length):
    def trace(values):
        x, angle, distance, focal = values
        source = Ray.numerical_type()(x=x, theta=angle, n=1.33)
        spaced = source.propagate(1.0, distance, 0.0, 1.0, n=source.n)
        return spaced.propagate(1.0, 0.0, -1.0/focal, 1.0, n=source.n).vector

    values = xp.array([1.2e-3, -0.018, 0.025, focal_length])
    x, angle, distance, focal = np.asarray(values)
    height = x + distance*angle
    expected_jacobian = np.array([
        [1.0, distance, angle, 0.0],
        [-1.0/focal, 1.0-distance/focal, -angle/focal, height/focal**2],
    ])
    height_hessian = np.zeros((4, 4))
    height_hessian[1, 2] = height_hessian[2, 1] = 1.0
    angle_hessian = np.array([
        [0.0, 0.0, 0.0, 1.0/focal**2],
        [0.0, 0.0, -1.0/focal, distance/focal**2],
        [0.0, -1.0/focal, 0.0, angle/focal**2],
        [1.0/focal**2, distance/focal**2, angle/focal**2, -2.0*height/focal**3],
    ])
    for function, expected in (
        (trace, [height, angle-height/focal]),
        (grad(lambda v: trace(v)[1]), expected_jacobian[1]),
        (jacobian(trace), expected_jacobian),
        (hessian(lambda v: trace(v)[0]), height_hessian),
        (hessian(lambda v: trace(v)[1]), angle_hessian),
    ):
        np.testing.assert_allclose(function(values), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jit(function)(values), expected, rtol=1e-12, atol=1e-12)
