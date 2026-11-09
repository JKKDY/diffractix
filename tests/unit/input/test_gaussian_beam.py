"""Gaussian input factories, physical domains, and backend-neutral derivatives."""

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


def test_gaussian_properties_follow_the_complex_q_definition_in_glass():
    wavelength, index = 1.064e-6, 1.5
    beam = GaussianBeam(q=0.03 + 0.04j, wavelength=wavelength, n=index).numeric()
    waist = np.sqrt(0.04 * wavelength / (np.pi * index))

    np.testing.assert_allclose(
        [beam.w, beam.R, beam.z_r, beam.z0, beam.w0, beam.divergence_angle,
         beam.NA, beam.gouy_phase, beam.b],
        [np.sqrt(wavelength / (16 * np.pi * index)), 1 / 12, 0.04, -0.03,
         waist, wavelength / (np.pi * index * waist),
         wavelength / (np.pi * waist), np.arctan2(0.03, 0.04), 0.08],
        rtol=1e-12,
    )
    assert beam.bindings["wavelength"] == wavelength


@pytest.mark.parametrize("distance", [-0.03, 0.0, 0.08])
def test_gaussian_free_space_helpers_match_analytical_beam_envelope(distance):
    width, wavelength, index = 0.7e-3, 1.064e-6, 1.33
    rayleigh_range = np.pi * index * width**2 / wavelength
    source = GaussianBeam.from_waist(width, wavelength, n=index).numeric()
    propagated = source.propagate(1.0, distance, 0.0, 1.0, n=index)
    expected_width = width * np.sqrt(1 + (distance / rayleigh_range)**2)

    assert propagated.q == pytest.approx(distance + 1j * rayleigh_range)
    assert propagated.w == pytest.approx(expected_width)
    assert source.w_at_z(distance) == pytest.approx(expected_width)
    if distance == 0:
        assert np.isinf(propagated.R) and np.isinf(source.R_at_z(distance))
    else:
        expected_radius = distance + rayleigh_range**2 / distance
        assert propagated.R == pytest.approx(expected_radius)
        assert source.R_at_z(distance) == pytest.approx(expected_radius)
    assert source.q == pytest.approx(1j * rayleigh_range)
    assert type(propagated) is type(source)


def test_overlap_matches_waist_size_formula_and_is_symmetric():
    first_width, second_width = 0.7e-3, 1.1e-3
    first = GaussianBeam.from_waist(first_width, 1.064e-6, n=1.5).numeric()
    second = GaussianBeam.from_waist(second_width, 1.064e-6, n=1.5).numeric()
    expected = 4 * first_width**2 * second_width**2 / (first_width**2 + second_width**2)**2

    assert first.overlap_with(first) == pytest.approx(1.0)
    assert first.overlap_with(second) == pytest.approx(expected)
    assert second.overlap_with(first) == pytest.approx(expected)
    assert 0 < expected < 1


def test_overlap_is_preserved_by_common_lossless_lens_and_drift():
    first = GaussianBeam(q=0.02 + 0.04j, wavelength=1.064e-6, n=1.33).numeric()
    second = GaussianBeam(q=-0.01 + 0.06j, wavelength=1.064e-6, n=1.33).numeric()
    distance, focal_length = 0.03, 0.12
    matrix = (1 - distance / focal_length, distance, -1 / focal_length, 1.0)

    propagated_first = first.propagate(*matrix, n=1.33)
    propagated_second = second.propagate(*matrix, n=1.33)

    assert propagated_first.overlap_with(propagated_second) == pytest.approx(
        first.overlap_with(second), rel=1e-12,
    )
    assert (first.q, second.q) == (0.02 + 0.04j, -0.01 + 0.06j)


def test_waist_envelope_derivatives_remain_finite_through_plane_curvature():
    width, wavelength = 0.7e-3, 1.064e-6
    rayleigh_range = np.pi * width**2 / wavelength
    beam = GaussianBeam.from_waist(width, wavelength).numeric()
    envelope = lambda distance: beam.w_at_z(distance)

    for function, expected in (
        (envelope, width),
        (grad(envelope), 0.0),
        (hessian(envelope), width / rayleigh_range**2),
    ):
        np.testing.assert_allclose(function(0.0), expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(jit(function)(0.0), expected, rtol=1e-12, atol=1e-15)



@pytest.mark.parametrize("q", [0, 1., -1j, 1-1j, complex(np.nan, 1), complex(0, np.nan),
                              complex(np.inf, 1), complex(0, np.inf)])
@pytest.mark.parametrize("entry", ["declarative", "numerical", "width"])
def test_invalid_q_is_rejected_before_division_or_width_evaluation(q, entry):
    with pytest.raises(ValueError, match="Invalid q-parameter"):
        if entry == "width":
            GaussianBeam.w_R_from_q(q, wavelength=1e-6, n=1)
        else:
            cls = GaussianBeam if entry == "declarative" else GaussianBeam.numerical_type()
            cls(q=q, wavelength=1e-6, n=1)


def test_q_validation_accepts_small_positive_rayleigh_range_without_epsilon():
    beam = GaussianBeam(q=1e-20j, wavelength=1e-6, n=1).numeric()
    width, radius = beam.w_R_from_q(beam.q, beam.wavelength, beam.n)
    assert width == pytest.approx(np.sqrt(1e-26/np.pi), rel=1e-12, abs=0)
    assert radius == np.inf


def test_q_validation_preserves_batched_values_and_traced_derivatives():
    cls = GaussianBeam.numerical_type()
    def width(t):
        return cls(q=t[0] + 1j*t[1], wavelength=1e-6, n=1).w
    point = xp.array([0.02, 0.03])
    reference = lambda t: xp.sqrt(1e-6/np.pi*(t[0]**2+t[1]**2)/t[1])
    for transform in (lambda f: f, grad, hessian):
        expected = transform(reference)(point)
        np.testing.assert_allclose(transform(width)(point), expected, rtol=1e-11)
        np.testing.assert_allclose(jit(transform(width))(point), expected, rtol=1e-11)
    q = xp.array([0.01+0.02j, 0.03j])
    beam = cls(q=q, wavelength=1e-6, n=1)
    np.testing.assert_allclose(beam.w, np.sqrt(-1e-6/(np.pi*np.imag(1/np.asarray(q)))))
    with pytest.raises(ValueError, match="Invalid q-parameter"):
        cls(q=xp.array([1j, -1j]), wavelength=1e-6, n=1)


@pytest.mark.parametrize("index_offset, compatible", [(5e-7, True), (2e-6, False)])
def test_overlap_medium_tolerance_is_one_part_per_million_and_symmetric(index_offset, compatible):
    first = GaussianBeam.from_waist(1e-3, 1e-6, n=1.0).numeric()
    second = GaussianBeam.from_waist(1e-3, 1e-6, n=1.0 + index_offset).numeric()
    for left, right in ((first, second), (second, first)):
        if compatible:
            assert left.overlap_with(right) == pytest.approx(1.0, abs=1e-6)
        else:
            with pytest.raises(ValueError, match="media"):
                left.overlap_with(right)


@pytest.mark.parametrize("wavelength", [1e-9, 1e-6, 1e-3])
@pytest.mark.parametrize("relative_difference,compatible", [(0.5e-6, True), (2e-6, False)])
def test_overlap_wavelength_tolerance_is_relative_and_symmetric(wavelength, relative_difference, compatible):
    cls = GaussianBeam.numerical_type()
    first = cls(q=1j, wavelength=wavelength, n=1)
    second = cls(q=1j, wavelength=wavelength*(1+relative_difference), n=1)
    for left, right in ((first, second), (second, first)):
        if compatible:
            assert left.overlap_with(right) == pytest.approx(1.)
        else:
            with pytest.raises(ValueError, match="wavelength"):
                left.overlap_with(right)


def test_overlap_wavelength_tolerance_accepts_float32_rounding():
    cls = GaussianBeam.numerical_type()
    first = cls(q=1j, wavelength=1.064e-6, n=1)
    rounded = cls(q=1j, wavelength=np.float32(1.064e-6), n=1)
    assert first.overlap_with(rounded) == pytest.approx(1.)
    assert rounded.overlap_with(first) == pytest.approx(1.)
