"""Vectorized traces agree with independently propagated rays."""
import numpy as np
import pytest
from diffractix import Interface, Parameter, Ray, RayBundle, Slab, Space, System, ThickLens, ThinLens
from diffractix.backend import grad, hessian, jacobian, jit, xp


@pytest.mark.parametrize("kind", ["space", "lens", "planar", "curved", "slab", "thick-lens", "chain"])
def test_bundle_trace_matches_independent_rays(kind):
    factories = {
        "space": lambda: (Space(d=0.025),), "lens": lambda: (ThinLens(f=0.12),),
        "planar": lambda: (Interface(n1=1.33, n2=1.52),),
        "curved": lambda: (Interface(n1=1.33, n2=1.52, R=0.08),),
        "slab": lambda: (Slab(d=0.008, n=1.52),),
        "thick-lens": lambda: (ThickLens(d=0.006, n=1.52, R1=0.08, R2=-0.11),),
        "chain": lambda: (Space(d=0.025), ThinLens(f=0.12), Slab(d=0.008, n=1.52), Interface(n1=1.33, n2=1.45), Space(d=0.01)),
    }
    elements = factories[kind]()
    x, angles = np.array([-0.0012, 0, 0.0009]), np.array([0.018, 0, -0.012])
    source = RayBundle(x=x, theta=angles, n=1.33)
    result = System(*elements, optical_input=source, ambient_n=1.33).build().run()
    references = [System(*elements, optical_input=Ray(x=a, theta=b, n=1.33), ambient_n=1.33).build().run() for a,b in zip(x,angles)]
    actual = xp.stack([state.vectors for state in result.states])
    expected = xp.stack([xp.stack([state.vector for state in ray.states]) for ray in references], axis=2)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    for reference in references:
        np.testing.assert_allclose([s.n for s in result.states], [s.n for s in reference.states])
        np.testing.assert_allclose(result.z, reference.z)
    np.testing.assert_array_equal(source.x, x)


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_bundle_systems(count):
    source = RayBundle(x=xp.array([0.001][:count]), theta=xp.array([0.01][:count]))
    result = System(Slab(d=0.01,n=1.5), optical_input=source).build().run()
    assert all(s.vectors.shape == (2,count) for s in result.states)
    assert len(result.states) == 4
    np.testing.assert_allclose(result.final.x, source.x+0.01*source.theta/1.5)


def test_bundle_derivatives_match_analytical_lens_chain():
    d, f = Parameter(0.02).variable(), Parameter(0.1).variable()
    x, angle = xp.array([-0.001, 0.002]), xp.array([0.01, -0.02])
    simulation = System(Space(d=d), ThinLens(f=f), optical_input=RayBundle(x=x, theta=angle)).build()
    mapping = simulation._optics_program.parameter_indices
    def actual(theta):
        return simulation.run(theta).final.vectors
    def expected(theta):
        height = x+theta[mapping[id(d)]]*angle
        return xp.stack([height, angle-height/theta[mapping[id(f)]]])
    theta = xp.array(simulation.initial_values.values)
    for transform in (lambda fun: fun, jacobian, lambda fun: grad(lambda t: xp.sum(fun(t)**2)), lambda fun: hessian(lambda t: xp.sum(fun(t)**2))):
        function, reference = transform(actual), transform(expected)
        np.testing.assert_allclose(function(theta), reference(theta), atol=1e-12)
        np.testing.assert_allclose(jit(function)(theta), reference(theta), atol=1e-12)


def test_converging_bundle_reaches_a_common_focus_through_the_system():
    focus = 0.08
    source = RayBundle.converging(radius=2e-3, count=7, focal_distance=focus)
    result = System(Space(d=focus), Space(d=0.02), optical_input=source).build().run()
    np.testing.assert_allclose(result.states[1].x, 0.0, atol=1e-15)
    np.testing.assert_allclose(result.final.x, -0.02 * source.x / focus, atol=1e-15)
    assert result.final.size == 7
