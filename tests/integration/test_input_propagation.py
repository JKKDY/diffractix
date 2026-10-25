"""Graph conversion, numerical optical states, and backend transformations compose."""

import numpy as np
import pytest

from diffractix import GaussianBeam, Parameter, Ray, RayBundle, Symbol
from diffractix.backend import grad, hessian, jacobian, jit, xp


@pytest.mark.parametrize("kind", ["ray", "bundle", "gaussian"])
def test_numerical_inputs_follow_an_analytic_abcd_chain_and_preserve_type(kind):
    x = np.array([1.2e-3, -0.9e-3]) if kind == "bundle" else 1.2e-3
    angle = np.array([-0.018, 0.012]) if kind == "bundle" else -0.018
    q = 0.01 + 0.03j
    wavelength = 1.064e-6
    if kind == "gaussian":
        declaration = GaussianBeam(q=q, wavelength=wavelength)
    elif kind == "bundle":
        declaration = RayBundle(x=xp.array(x), theta=xp.array(angle))
    else:
        declaration = Ray(x=x, theta=angle)
    source = declaration.numeric({"ambient_n": 1.0})
    distance, focal, output_index = 0.025, 0.12, 1.5

    spaced = source.propagate(1.0, distance, 0.0, 1.0, n=1.0)
    focused = spaced.propagate(1.0, 0.0, -1.0 / focal, 1.0, n=1.0)
    final = focused.propagate(1.0, 0.0, 0.0, 1.0 / output_index, n=output_index)

    assert all(type(state) is declaration.numerical_type() for state in (source, spaced, focused, final))
    assert final.parameter_names == ()
    assert final.refractive_index == output_index
    assert source.refractive_index == 1.0
    if kind == "gaussian":
        expected_q = output_index * (q + distance) / (1 - (q + distance) / focal)
        np.testing.assert_allclose(final.q, expected_q, rtol=1e-13)
        expected_width = np.sqrt(-wavelength / (np.pi * output_index * np.imag(1 / expected_q)))
        np.testing.assert_allclose(final.w, expected_width, rtol=1e-13)
        assert source.q == q
        assert final.wavelength == wavelength
    else:
        expected_height = x + distance * angle
        expected_angle = (angle - expected_height / focal) / output_index
        np.testing.assert_allclose(final.x, expected_height, rtol=1e-13)
        np.testing.assert_allclose(final.theta, expected_angle, rtol=1e-13)
        np.testing.assert_array_equal(source.x, x)
        np.testing.assert_array_equal(source.theta, angle)


def test_direct_numerical_ray_construction_and_assignment_are_differentiable_and_jittable():
    numerical_ray = Ray.numerical_type()

    def trace(values):
        x, angle, distance, focal = values
        source = numerical_ray(x=x, theta=0.0, n=1.33)
        source.theta = angle
        spaced = source.propagate(1.0, distance, 0.0, 1.0, n=1.33)
        return spaced.propagate(1.0, 0.0, -1.0 / focal, 1.0, n=1.33).vector

    point = xp.array([1.2e-3, -0.018, 0.025, 0.12])
    x, angle, distance, focal = np.asarray(point)
    height = x + distance * angle
    expected_jacobian = np.array([
        [1.0, distance, angle, 0.0],
        [-1.0 / focal, 1 - distance / focal, -angle / focal, height / focal**2],
    ])
    expected_hessian = np.array([
        [0.0, 0.0, 0.0, 1.0 / focal**2],
        [0.0, 0.0, -1.0 / focal, distance / focal**2],
        [0.0, -1.0 / focal, 0.0, angle / focal**2],
        [1.0 / focal**2, distance / focal**2, angle / focal**2, -2 * height / focal**3],
    ])
    for function, expected in (
        (trace, [height, angle - height / focal]),
        (grad(lambda values: trace(values)[1]), expected_jacobian[1]),
        (jacobian(trace), expected_jacobian),
        (hessian(lambda values: trace(values)[1]), expected_hessian),
    ):
        np.testing.assert_allclose(function(point), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jit(function)(point), expected, rtol=1e-12, atol=1e-12)


def test_direct_numerical_bundle_vector_derivatives_match_independent_rays():
    numerical_bundle = RayBundle.numerical_type()
    positions = xp.array([-1.2e-3, 0.0, 0.9e-3])
    angles = xp.array([0.018, 0.0, -0.012])

    def trace(distance):
        source = numerical_bundle(x=positions, theta=angles, n=1.33)
        return source.propagate(1.0, distance, 0.0, 1.0, n=1.33).vectors

    def cost(distance):
        return xp.sum(trace(distance)[0] ** 2)

    distance = 0.025
    expected = np.stack((np.asarray(positions + distance * angles), np.asarray(angles)))
    expected_jacobian = np.stack((np.asarray(angles), np.zeros(3)))
    expected_gradient = 2 * np.sum(np.asarray(angles) * expected[0])
    expected_hessian = 2 * np.sum(np.asarray(angles) ** 2)
    for function, answer in (
        (trace, expected),
        (jacobian(trace), expected_jacobian),
        (grad(cost), expected_gradient),
        (hessian(cost), expected_hessian),
    ):
        np.testing.assert_allclose(function(distance), answer, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(jit(function)(distance), answer, rtol=1e-12, atol=1e-15)


def test_numerical_gaussian_factory_and_complex_propagation_derivatives():
    numerical_gaussian = GaussianBeam.numerical_type()
    width, wavelength, distance = 0.5e-3, 1e-6, 0.04
    q0 = 1j * np.pi * width**2 / wavelength

    def trace(focal):
        source = numerical_gaussian.from_waist(width, wavelength, n=1.0)
        focused = source.propagate(1.0, 0.0, -1.0 / focal, 1.0, n=1.0)
        propagated = focused.propagate(1.0, distance, 0.0, 1.0, n=1.0)
        return xp.stack((xp.real(propagated.q), xp.imag(propagated.q)))

    def cost(focal):
        return xp.sum(trace(focal) ** 2)

    focal = 0.12
    q = focal * q0 / (focal - q0) + distance
    first = -q0**2 / (focal - q0) ** 2
    second = 2 * q0**2 / (focal - q0) ** 3
    for function, expected in (
        (trace, [q.real, q.imag]),
        (jacobian(trace), [first.real, first.imag]),
        (grad(cost), 2 * np.real(np.conj(q) * first)),
        (hessian(cost), 2 * abs(first) ** 2 + 2 * np.real(np.conj(q) * second)),
    ):
        np.testing.assert_allclose(function(focal), expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(jit(function)(focal), expected, rtol=1e-12, atol=1e-14)


def test_bindings_through_numeric_and_propagation_keep_nested_derivatives():
    angle = Parameter(0.02).variable()
    declaration = Ray(x=Symbol("position"), theta=angle)
    output_index = 1.5

    def trace(values):
        position, index, distance = values
        source = declaration.numeric({"position": position, "ambient_n": index})
        refracted = source.propagate(1.0, 0.0, 0.0, index / output_index, n=output_index)
        return refracted.propagate(1.0, distance, 0.0, 1.0, n=output_index).vector

    point = xp.array([1e-3, 1.2, 0.025])
    position, index, distance = np.asarray(point)
    coefficient = angle.value / output_index
    expected = [position + distance * index * coefficient, index * coefficient]
    expected_jacobian = np.array([
        [1.0, distance * coefficient, index * coefficient],
        [0.0, coefficient, 0.0],
    ])
    expected_hessian = np.zeros((3, 3))
    expected_hessian[1, 2] = expected_hessian[2, 1] = coefficient
    nested = jacobian(grad(lambda values: trace(values)[0]))
    for function, answer in (
        (trace, expected),
        (jacobian(trace), expected_jacobian),
        (hessian(lambda values: trace(values)[0]), expected_hessian),
        (nested, expected_hessian),
    ):
        np.testing.assert_allclose(function(point), answer, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(jit(function)(point), answer, rtol=1e-12, atol=1e-14)
    assert angle.is_variable
    assert isinstance(declaration.x.node, Symbol)
