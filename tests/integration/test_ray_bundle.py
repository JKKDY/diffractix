"""RayBundle systems must match independent Ray traces and derivatives."""

import numpy as np
import pytest

from diffractix import Interface, Parameter, Ray, RayBundle, Slab, Space, System, ThickLens, ThinLens
from diffractix.backend import grad, hessian, jacobian, jit, xp


@pytest.fixture
def source():
    return RayBundle(
        x=xp.array([-1.2e-3, 0.0, 0.9e-3, 2e-3]),
        theta=xp.array([0.018, 0.0, -0.012, 0.006]),
        n=1.33,
    ).numeric()


@pytest.mark.parametrize("kind", ["space", "lens", "planar", "curved", "slab", "thick-lens", "chain"])
def test_bundle_optical_traces_match_independent_rays(source, kind):
    factories = {
        "space": lambda: (Space(d=0.025),),
        "lens": lambda: (ThinLens(f=0.12),),
        "planar": lambda: (Interface(n1=source.n, n2=1.52),),
        "curved": lambda: (Interface(n1=source.n, n2=1.52, R=0.08),),
        "slab": lambda: (Slab(d=0.008, n=1.52),),
        "thick-lens": lambda: (ThickLens(d=0.006, n=1.52, R1=0.08, R2=-0.11),),
        "chain": lambda: (
            Space(d=0.025), ThinLens(f=0.12), Slab(d=0.008, n=1.52),
            ThickLens(d=0.006, n=1.6, R1=0.08, R2=-0.11),
            Interface(n1=source.n, n2=1.45, R=-0.1), Space(d=0.01, n=1.45),
        ),
    }
    elements = factories[kind]()
    original = np.array(source.vectors, copy=True)
    result = System(*elements, input=source, ambient_n=source.n).build().run()
    references = [
        System(*elements, input=Ray.numerical_type()(x=x, theta=angle, n=source.n), ambient_n=source.n).build().run()
        for x, angle in zip(source.x, source.theta)
    ]

    assert all(isinstance(state, RayBundle) and state.size == source.size for state in result.states)
    assert result.x.shape == result.theta.shape == (len(result.states), source.size)
    assert result.vectors.shape == (len(result.states), 2, source.size)
    np.testing.assert_allclose(result.x, np.stack([ray.x for ray in references], axis=1), atol=1e-15)
    np.testing.assert_allclose(result.theta, np.stack([ray.theta for ray in references], axis=1), atol=1e-15)
    for ray in references:
        np.testing.assert_allclose(result.n, ray.n, atol=1e-15)
        np.testing.assert_allclose(result.z, ray.z, atol=1e-15)
    np.testing.assert_array_equal(source.vectors, original)
    assert source.n == 1.33


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_bundles_propagate_through_composites(count):
    source = RayBundle(x=xp.array([1.2e-3][:count]), theta=xp.array([-0.018][:count])).numeric({"ambient_n": 1.0})
    elements = (Space(d=0.025), ThinLens(f=0.12), Slab(d=0.008, n=1.52))
    result = System(*elements, input=source).build().run()

    assert len(result.states) == 6
    assert result.x.shape == result.theta.shape == (6, count)
    assert result.vectors.shape == (6, 2, count)
    assert all(state.size == count for state in result.states)
    np.testing.assert_allclose(result.n, [1.0, 1.0, 1.0, 1.52, 1.52, 1.0], atol=1e-15)
    if count:
        reference = System(*elements, input=Ray(x=1.2e-3, theta=-0.018).numeric({"ambient_n": 1.0})).build().run()
        np.testing.assert_allclose(result.x[:, 0], reference.x, atol=1e-15)
        np.testing.assert_allclose(result.theta[:, 0], reference.theta, atol=1e-15)


def test_bundle_simulation_derivatives_and_jit_match_independent_rays(source):
    distance = Parameter(0.025).variable(bounds=(0.01, 0.06))
    focal = Parameter(0.12).variable(bounds=(0.08, 0.2))
    elements = (Space(d=distance), ThinLens(f=focal), Slab(d=0.008, n=1.52))
    simulation = System(*elements, input=source, ambient_n=source.n).build()
    references = [
        System(*elements, input=Ray.numerical_type()(x=x, theta=angle, n=source.n), ambient_n=source.n).build()
        for x, angle in zip(source.x, source.theta)
    ]
    assert simulation.graph.variables == (distance, focal)
    assert all(reference.graph.variables == (distance, focal) for reference in references)

    def bundle_values(values):
        return simulation.run(values).final.vectors

    def ray_values(values):
        return xp.stack([reference.run(values).final.vector for reference in references], axis=1)

    bundle_cost = lambda v: xp.sum(xp.square(bundle_values(v)))
    ray_cost = lambda v: xp.sum(xp.square(ray_values(v)))
    for function, reference, shape in (
        (bundle_values, ray_values, (2, source.size)),
        (jacobian(bundle_values), jacobian(ray_values), (2, source.size, 2)),
        (grad(bundle_cost), grad(ray_cost), (2,)),
        (hessian(bundle_cost), hessian(ray_cost), (2, 2)),
    ):
        compiled = jit(function)
        for point in ([0.025, 0.12], [0.035, 0.16]):
            values = xp.array(point)
            expected = reference(values)
            eager, jitted = function(values), compiled(values)
            assert np.shape(eager) == np.shape(jitted) == shape
            np.testing.assert_allclose(eager, expected, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(jitted, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(simulation.initial_values, [0.025, 0.12])
    assert (distance.value, focal.value) == (0.025, 0.12)


@pytest.mark.parametrize("kind", ["parallel", "converging"])
def test_factory_bundles_propagate_and_expose_stacked_statistics(kind):
    if kind == "parallel":
        source = RayBundle.parallel(radius=2e-3, count=5, n=1.33).numeric()
        expected_x = [-2e-3, -1e-3, 0.0, 1e-3, 2e-3]
        expected_spread = np.sqrt(2)*1e-3
    else:
        source = RayBundle.converging(radius=2e-3, count=5, focal_distance=0.05, n=1.33).numeric()
        expected_x = np.zeros(5)
        expected_spread = 0.0
    result = System(Space(d=0.05), input=source, ambient_n=source.n).build().run()

    np.testing.assert_allclose(result.final.x, expected_x, atol=1e-15)
    np.testing.assert_allclose(result.final.theta, source.theta, atol=1e-15)
    np.testing.assert_allclose(result.n, [1.33, 1.33], atol=1e-15)
    np.testing.assert_allclose(result.centroid, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(result.mean_angle, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(result.position_spread, [np.sqrt(2)*1e-3, expected_spread], atol=1e-15)
    np.testing.assert_allclose(result.rms_x, [np.sqrt(2)*1e-3, expected_spread], atol=1e-15)
    expected_angular_spread = 0.0 if kind == "parallel" else np.sqrt(2)*1e-3/0.05
    np.testing.assert_allclose(result.angular_spread, [expected_angular_spread]*2, atol=1e-15)
