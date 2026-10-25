"""Gaussian factory domains and backend-neutral differentiation."""

import numpy as np
import pytest

from diffractix import GaussianBeam
from diffractix.backend import grad, hessian, jacobian, jit, xp


FACTORIES = (
    (GaussianBeam.from_waist, {"w0": 1e-3}),
    (GaussianBeam.from_w_and_R, {"w": 1e-3, "R": np.inf}),
    (GaussianBeam.from_divergence, {"theta": 0.01}),
    (GaussianBeam.from_fiber_tip, {"NA": 0.1}),
    (GaussianBeam.from_fiber_tip, {"MFD": 2e-3}),
)


@pytest.mark.parametrize("factory, arguments", FACTORIES, ids=["waist", "wavefront", "divergence", "NA", "MFD"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_factory_rejects_nonpositive_or_nonfinite_size(factory, arguments, value):
    arguments = dict(arguments)
    name = next(iter(arguments))
    arguments[name] = value
    with pytest.raises(ValueError, match=rf"{name}.*finite.*positive"):
        factory(wavelength=1e-6, **arguments)


@pytest.mark.parametrize("name", ["wavelength", "n"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_factories_validate_wavelength_and_medium_before_computation(name, value):
    for factory, arguments in FACTORIES:
        inputs = dict(arguments, wavelength=1e-6, n=1.5)
        inputs[name] = value
        with pytest.raises(ValueError, match=rf"{name}.*finite.*positive"):
            factory(**inputs)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1j, "0.001", object()])
def test_factory_rejects_non_real_size(value):
    with pytest.raises(TypeError, match="w0.*real"):
        GaussianBeam.from_waist(value, wavelength=1e-6)


def test_factories_describe_the_same_waist_in_a_non_air_medium():
    width, wavelength, index = 8e-4, 1.064e-6, 1.5
    beams = (
        GaussianBeam.from_waist(width, wavelength, n=index),
        GaussianBeam.from_w_and_R(width, np.inf, wavelength, n=index),
        GaussianBeam.from_divergence(wavelength / (np.pi * index * width), wavelength, n=index),
        GaussianBeam.from_fiber_tip(wavelength, NA=wavelength / (np.pi * width), n=index),
        GaussianBeam.from_fiber_tip(wavelength, MFD=2 * width, n=index),
    )
    expected_q = 1j * np.pi * width**2 * index / wavelength
    for declaration in beams:
        beam = declaration.numeric()
        assert beam.q == pytest.approx(expected_q)
        assert beam.w == pytest.approx(width)
        assert beam.n == index


@pytest.mark.parametrize("radius", [0.2, -0.2, np.inf, -np.inf])
def test_wavefront_radius_allows_both_signs_and_plane_limit(radius):
    beam = GaussianBeam.from_w_and_R(1e-3, radius, 1e-6)
    expected_inverse_q = (0.0 if np.isinf(radius) else 1.0 / radius) - 1j / np.pi
    assert 1 / beam.q == pytest.approx(expected_inverse_q)


@pytest.mark.parametrize("radius", [0.0, np.nan])
def test_wavefront_radius_rejects_singular_or_nan_values(radius):
    with pytest.raises(ValueError, match="R"):
        GaussianBeam.from_w_and_R(1e-3, radius, 1e-6)


@pytest.mark.parametrize("position", [np.nan, np.inf, -np.inf])
def test_waist_position_must_be_finite(position):
    with pytest.raises(ValueError, match="z_waist_loc.*finite"):
        GaussianBeam.from_waist(1e-3, 1e-6, z_waist_loc=position)


def test_fiber_mode_selection_retains_na_precedence_and_requires_a_mode():
    with pytest.raises(ValueError, match="NA.*MFD"):
        GaussianBeam.from_fiber_tip(1e-6)
    beam = GaussianBeam.from_fiber_tip(1e-6, NA=0.1, MFD=50e-6).numeric()
    assert beam.w == pytest.approx(1e-6 / (np.pi * 0.1))


def test_factory_preserves_backend_arrays_and_broadcasting():
    widths = xp.array([5e-4, 1e-3], dtype="float32")
    beam = GaussianBeam.numerical_type().from_waist(widths, wavelength=1e-6)
    np.testing.assert_allclose(beam.w, widths, rtol=1e-6)
    assert beam.q.shape == (2,)
    with pytest.raises(ValueError, match="w0.*positive"):
        GaussianBeam.from_waist(xp.array([5e-4, -1e-3]), wavelength=1e-6)


def test_waist_factory_first_second_and_nested_derivatives_and_jit():
    def moments(t):
        beam = GaussianBeam.numerical_type().from_waist(1e-3 * t[0], 1e-6 * t[1], n=t[2],
                                       z_waist_loc=0.01 * t[3])
        return xp.stack((xp.real(beam.q) / 0.01, xp.imag(beam.q) / np.pi))

    point = xp.array([1.2, 0.9, 1.5, -2.0])
    a, b, c, z = np.asarray(point)
    expected_gradient = [2*a*c/b, -a*a*c/b**2, a*a/b, 0]
    expected_hessian = np.array([
        [2*c/b, -2*a*c/b**2, 2*a/b, 0],
        [-2*a*c/b**2, 2*a*a*c/b**3, -a*a/b**2, 0],
        [2*a/b, -a*a/b**2, 0, 0],
        [0, 0, 0, 0],
    ])
    f = lambda t: moments(t)[1]
    reference = lambda t: t[0]**2 * t[2] / t[1]
    nested = lambda function: hessian(lambda t: xp.sum(grad(function)(t)**2))
    for function, expected in (
        (moments, [-z, a*a*c/b]),
        (jacobian(moments), [[0, 0, 0, -1], expected_gradient]),
        (grad(f), expected_gradient),
        (hessian(f), expected_hessian),
        (nested(f), nested(reference)(point)),
    ):
        np.testing.assert_allclose(function(point), expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(jit(function)(point), expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kind", ["wavefront", "divergence", "NA", "MFD"])
def test_other_factory_derivatives_and_jit_match_independent_formulas(kind):
    def imaginary_q(t):
        a, wavelength, n, radius = t
        if kind == "wavefront":
            beam = GaussianBeam.numerical_type().from_w_and_R(a, radius, wavelength, n=n)
        elif kind == "divergence":
            beam = GaussianBeam.numerical_type().from_divergence(a, wavelength, n=n)
        elif kind == "NA":
            beam = GaussianBeam.numerical_type().from_fiber_tip(wavelength, NA=a, n=n)
        else:
            beam = GaussianBeam.numerical_type().from_fiber_tip(wavelength, MFD=a, n=n)
        return xp.imag(beam.q)

    def reference(t):
        a, wavelength, n, radius = t
        if kind == "wavefront":
            return xp.imag(1 / (1/radius - 1j*wavelength/(np.pi*n*a**2)))
        if kind == "divergence":
            return wavelength / (np.pi*n*a**2)
        if kind == "NA":
            return wavelength*n / (np.pi*a**2)
        return np.pi*(a/2)**2*n/wavelength

    point = xp.array([0.3, 0.8, 1.5, -0.2])
    for differentiate in (lambda f: f, grad, hessian):
        expected = differentiate(reference)(point)
        function = differentiate(imaginary_q)
        np.testing.assert_allclose(function(point), expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(jit(function)(point), expected, rtol=1e-11, atol=1e-12)
