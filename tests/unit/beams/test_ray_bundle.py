"""Vectorized independent-ray propagation, shape contracts, and derivatives."""

import numpy as np
import pytest

from diffractix import Ray, RayBundle
from diffractix.backend import grad, hessian, jacobian, jit, xp


@pytest.fixture
def bundle():
    return RayBundle(
        x=xp.array([-1.2e-3, 0.0, 0.9e-3, 2.0e-3]),
        theta=xp.array([0.018, 0.0, -0.012, 0.006]),
        n=1.33,
    )


def test_bundle_construction_default_medium_size_and_vectors():
    positions = xp.array([-1.2e-3, 0.0, 0.9e-3])
    angles = xp.array([0.018, 0.0, -0.012])
    bundle = RayBundle(x=positions, theta=angles)

    assert bundle.n == 1.0
    assert bundle.size == 3
    assert bundle.vectors.shape == (2, 3)
    np.testing.assert_array_equal(bundle.vectors, [positions, angles])


def test_bundle_x_at_z_extrapolates_every_ray_without_changing_source(bundle):
    original = np.array(bundle.vectors, copy=True)

    np.testing.assert_allclose(bundle.x_at_z(0.025), original[0] + 0.025*original[1], atol=1e-15)
    np.testing.assert_array_equal(bundle.vectors, original)
    assert bundle.n == 1.33


@pytest.mark.parametrize("positions, angles", [([], []), ([1.2e-3], [-0.018])])
def test_empty_and_single_ray_bundles_preserve_the_ray_axis(positions, angles):
    bundle = RayBundle(x=xp.array(positions), theta=xp.array(angles))
    count = len(positions)
    propagated = bundle.propagate(1.0, 0.0, 0.0, 1.0/1.52, n=1.52)

    assert bundle.size == propagated.size == count
    assert bundle.vectors.shape == propagated.vectors.shape == (2, count)
    assert bundle.x_at_z(0.025).shape == (count,)
    np.testing.assert_allclose(propagated.x, positions, atol=1e-15)
    np.testing.assert_allclose(propagated.theta, np.asarray(angles)/1.52, atol=1e-15)
    assert propagated.n == 1.52
    assert bundle.n == 1.0
    if count:
        scalar = Ray(x=positions[0], theta=angles[0]).propagate(1, 0, 0, 1/1.52, n=1.52)
        np.testing.assert_allclose(propagated.vectors[:, 0], scalar.vector, atol=1e-15)
        assert bundle.centroid == pytest.approx(positions[0])
        assert bundle.mean_angle == pytest.approx(angles[0])
        assert bundle.position_spread == bundle.angular_spread == 0.0
        assert bundle.rms_x == pytest.approx(abs(positions[0]))


@pytest.mark.parametrize("positions, angles, message", [
    ([0.0, 1e-3], [0.01], "matching shapes"),
    ([0.0, 1e-3], [[0.01, 0.02]], "matching shapes"),
    ([[0.0, 1e-3]], [[0.01, 0.02]], "one-dimensional"),
    (1e-3, 0.01, "one-dimensional"),
])
def test_bundle_rejects_mismatched_or_non_vector_inputs(positions, angles, message):
    with pytest.raises(ValueError, match=message):
        RayBundle(x=xp.array(positions), theta=xp.array(angles))


@pytest.mark.parametrize("matrix, output_index", [
    pytest.param((1, 0.025, 0, 1), 1.33, id="space"),
    pytest.param((1, 0, -1/0.12, 1), 1.33, id="converging-lens"),
    pytest.param((1, 0, 1/0.12, 1), 1.33, id="diverging-lens"),
    pytest.param((1, 0, 0, 1.33/1.52), 1.52, id="planar-interface"),
    pytest.param((1, 0, (1.33-1.52)/(0.08*1.52), 1.33/1.52), 1.52, id="curved-interface"),
    pytest.param((1, 0, (1.33-1.52)/(-0.08*1.52), 1.33/1.52), 1.52, id="negative-curvature"),
])
def test_bundle_abcd_matches_independent_rays_and_does_not_mutate_source(bundle, matrix, output_index):
    original = np.array(bundle.vectors, copy=True)
    rays = [Ray(x=x, theta=angle, n=bundle.n) for x, angle in zip(bundle.x, bundle.theta)]
    scalar_results = [ray.propagate(*matrix, n=output_index) for ray in rays]
    propagated = bundle.propagate(*matrix, n=output_index)

    assert isinstance(propagated, RayBundle)
    assert propagated is not bundle
    assert propagated.vectors.shape == (2, bundle.size)
    np.testing.assert_allclose(
        propagated.vectors, xp.stack([ray.vector for ray in scalar_results], axis=1),
        rtol=1e-13, atol=1e-15,
    )
    assert all(ray.n == propagated.n == output_index for ray in scalar_results)
    np.testing.assert_array_equal(bundle.vectors, original)
    assert bundle.n == 1.33


def test_sequential_bundle_propagation_matches_rays_and_matrix_product(bundle):
    source_vectors = np.array(bundle.vectors, copy=True)
    rays = [Ray(x=x, theta=angle, n=bundle.n) for x, angle in zip(bundle.x, bundle.theta)]
    combined = np.eye(2)
    sequence = (
        ((1, 0.025, 0, 1), 1.33),
        ((1, 0, -1/0.12, 1), 1.33),
        ((1, 0, (1.33-1.52)/(0.08*1.52), 1.33/1.52), 1.52),
        ((1, 0.01, 0, 1), 1.52),
        ((1, 0, 0, 1.52), 1.0),
    )
    for matrix, output_index in sequence:
        bundle = bundle.propagate(*matrix, n=output_index)
        rays = [ray.propagate(*matrix, n=output_index) for ray in rays]
        combined = np.reshape(matrix, (2, 2)) @ combined
        np.testing.assert_allclose(bundle.vectors, xp.stack([ray.vector for ray in rays], axis=1), atol=1e-15)
        assert all(ray.n == bundle.n == output_index for ray in rays)
    np.testing.assert_allclose(bundle.vectors, combined @ source_vectors, rtol=1e-13, atol=1e-15)


def test_bundle_source_and_optical_derivatives_have_independent_ray_axes():
    def trace(values):
        distance, focal, selected_x, selected_angle = values
        source = RayBundle(
            x=xp.stack((-1.2e-3, selected_x, 2e-3)),
            theta=xp.stack((0.018, 0.0, selected_angle)),
            n=1.33,
        )
        spaced = source.propagate(1, distance, 0, 1, n=source.n)
        return spaced.propagate(1, 0, -1/focal, 1, n=source.n).vectors

    values = xp.array([0.025, 0.12, 0.9e-3, -0.012])
    distance, focal, selected_x, selected_angle = np.asarray(values)
    positions = np.array([-1.2e-3, selected_x, 2e-3])
    angles = np.array([0.018, 0.0, selected_angle])
    height = positions + distance*angles
    select_x, select_angle = np.array([0, 1, 0]), np.array([0, 0, 1])
    expected_jacobian = np.stack((
        np.column_stack((angles, np.zeros(3), select_x, distance*select_angle)),
        np.column_stack((-angles/focal, height/focal**2, -select_x/focal, (1-distance/focal)*select_angle)),
    ))
    expected_hessian = np.array([
        [0, angles.sum()/focal**2, 0, -1/focal],
        [angles.sum()/focal**2, -2*height.sum()/focal**3, 1/focal**2, distance/focal**2],
        [0, 1/focal**2, 0, 0],
        [-1/focal, distance/focal**2, 0, 0],
    ])
    total_angle = lambda v: xp.sum(trace(v)[1])
    for function, expected in (
        (trace, np.stack((height, angles-height/focal))),
        (jacobian(trace), expected_jacobian),
        (grad(total_angle), expected_jacobian[1].sum(axis=0)),
        (hessian(total_angle), expected_hessian),
    ):
        eager, compiled = function(values), jit(function)(values)
        assert np.shape(eager) == np.shape(compiled) == np.shape(expected)
        np.testing.assert_allclose(eager, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(compiled, expected, rtol=1e-12, atol=1e-12)


def test_parallel_factory_samples_aperture_and_preserves_medium():
    bundle = RayBundle.parallel(radius=2e-3, count=5, n=1.33)

    assert bundle.size == 5
    assert bundle.n == 1.33
    np.testing.assert_allclose(bundle.x, [-2e-3, -1e-3, 0.0, 1e-3, 2e-3], atol=1e-15)
    np.testing.assert_array_equal(bundle.theta, np.zeros(5))


def test_converging_factory_points_all_rays_toward_common_focus():
    bundle = RayBundle.converging(radius=2e-3, count=5, focal_distance=0.05)

    assert bundle.size == 5
    assert bundle.n == 1.0
    np.testing.assert_allclose(bundle.x, [-2e-3, -1e-3, 0.0, 1e-3, 2e-3], atol=1e-15)
    np.testing.assert_allclose(bundle.theta, [0.04, 0.02, 0.0, -0.02, -0.04], atol=1e-15)
    np.testing.assert_allclose(bundle.x_at_z(0.05), np.zeros(5), atol=1e-15)


def test_bundle_statistics_distinguish_spread_from_axis_referenced_rms():
    bundle = RayBundle(x=xp.array([1e-3, 2e-3, 4e-3]), theta=xp.array([0.01, 0.02, 0.04]))

    assert bundle.centroid == pytest.approx(7e-3/3)
    assert bundle.mean_angle == pytest.approx(0.07/3)
    assert bundle.position_spread == pytest.approx(np.sqrt(14)*1e-3/3)
    assert bundle.angular_spread == pytest.approx(np.sqrt(14)*0.01/3)
    assert bundle.rms_x == pytest.approx(np.sqrt(7)*1e-3)


def test_bundle_statistics_derivatives_and_jit_away_from_zero_spread():
    def metrics(values):
        bundle = RayBundle(x=values[:3], theta=values[3:])
        return xp.stack((bundle.centroid, bundle.mean_angle, bundle.position_spread,
                         bundle.angular_spread, bundle.rms_x))

    values = xp.array([1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.04])
    positions, angles = np.asarray(values[:3]), np.asarray(values[3:])
    centered_x, centered_angles = positions-positions.mean(), angles-angles.mean()
    spread_x, spread_angle = np.linalg.norm(centered_x)/np.sqrt(3), np.linalg.norm(centered_angles)/np.sqrt(3)
    rms = np.linalg.norm(positions)/np.sqrt(3)
    expected_jacobian = np.zeros((5, 6))
    expected_jacobian[0, :3] = expected_jacobian[1, 3:] = 1/3
    expected_jacobian[2, :3] = centered_x/(3*spread_x)
    expected_jacobian[3, 3:] = centered_angles/(3*spread_angle)
    expected_jacobian[4, :3] = positions/(3*rms)
    expected_hessian = np.zeros((6, 6))
    expected_hessian[:3, :3] = np.eye(3)/(3*rms) - np.outer(positions, positions)/(9*rms**3)
    for function, expected in (
        (metrics, [positions.mean(), angles.mean(), spread_x, spread_angle, rms]),
        (jacobian(metrics), expected_jacobian),
        (grad(lambda v: metrics(v)[4]), expected_jacobian[4]),
        (hessian(lambda v: metrics(v)[4]), expected_hessian),
    ):
        np.testing.assert_allclose(function(values), expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(jit(function)(values), expected, rtol=1e-11, atol=1e-12)


def test_converging_factory_radius_and_focus_derivatives_and_jit():
    def trace(values):
        return RayBundle.converging(radius=values[0], count=5, focal_distance=values[1]).vectors

    values = xp.array([2e-3, 0.05])
    radius, focal = np.asarray(values)
    grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected_jacobian = np.stack((
        np.column_stack((grid, np.zeros(5))),
        np.column_stack((-grid/focal, radius*grid/focal**2)),
    ))
    angular_energy = lambda v: xp.sum(xp.square(trace(v)[1]))
    expected_hessian = np.array([
        [5/focal**2, -10*radius/focal**3],
        [-10*radius/focal**3, 15*radius**2/focal**4],
    ])
    for function, expected in (
        (trace, np.stack((radius*grid, -radius*grid/focal))),
        (jacobian(trace), expected_jacobian),
        (grad(angular_energy), [5*radius/focal**2, -5*radius**2/focal**3]),
        (hessian(angular_energy), expected_hessian),
    ):
        np.testing.assert_allclose(function(values), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jit(function)(values), expected, rtol=1e-12, atol=1e-12)
