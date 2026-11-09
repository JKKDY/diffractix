"""Physical-angle ray traces through compiled optical elements."""
import numpy as np
import pytest
from diffractix import FourF, Interface, Mirror, Parameter, Ray, Slab, Space, System, Telescope, ThickLens, ThinLens
from diffractix.backend import jacobian, hessian, jit, xp


@pytest.mark.parametrize("f", [0.12, -0.12])
def test_space_lens_space_analytical_trace(f):
    first, lens, last = Space(d=0.025), ThinLens(f=f), Space(d=0.015)
    result = System(first, lens, last, optical_input=Ray(x=0.0012, theta=0.018, n=1.33)).build().run()
    height = 0.0012 + 0.025 * 0.018
    angle = 0.018 - height/f
    np.testing.assert_allclose([s.x for s in result.states], [0.0012, height, height, height+0.015*angle])
    np.testing.assert_allclose([s.theta for s in result.states], [0.018, 0.018, angle, angle])
    np.testing.assert_allclose([s.n for s in result.states], [1.33]*4)
    np.testing.assert_allclose(result.z, [0, 0.025, 0.025, 0.04])
    assert result[lens].before is result.states[1]
    assert result[lens].after is result.states[2]


@pytest.mark.parametrize("radius", [np.inf, 0.08, -0.08])
def test_interface_physical_angle_and_index(radius):
    interface = Interface(n1=1.33, n2=1.52, R=radius)
    result = System(interface, optical_input=Ray(x=0.0012, theta=0.018, n=1.33)).build().run()
    expected = (1.33*0.018 - (1.52-1.33)*0.0012/radius)/1.52
    np.testing.assert_allclose(result.final.vector, [0.0012, expected])
    assert result[interface].before.n == pytest.approx(1.33)
    assert result[interface].after.n == pytest.approx(1.52)
    assert result[interface].z_start == result[interface].z_end == 0


@pytest.mark.parametrize("curved", [False, True])
def test_composite_matches_two_refractions_and_physical_translation(curved):
    d, n, r1, r2 = 0.006, 1.52, 0.08, -0.11
    element = ThickLens(d=d, n=n, R1=r1, R2=r2) if curved else Slab(d=d, n=n)
    result = System(element, optical_input=Ray(x=0.0012, theta=0.018, n=1.33), ambient_n=1.33).build().run()
    inside_angle = (1.33*0.018 - (n-1.33)*0.0012/r1)/n if curved else 1.33*0.018/n
    height = 0.0012+d*inside_angle
    exit_angle = (n*inside_angle - (1.33-n)*height/r2)/1.33 if curved else 0.018
    np.testing.assert_allclose(result.final.vector, [height, exit_angle])
    np.testing.assert_allclose([s.n for s in result.states], [1.33, n, n, 1.33])
    assert result[element.front].before is result.initial
    assert result[element.back].after is result.final


def test_planar_interface_index_derivatives_remain_finite_under_jit():
    n = Parameter(1.5).variable(bounds=(1.1, 2))
    simulation = System(Interface(n1=1, n2=n), Space(d=0.02), optical_input=Ray(x=0.0012, theta=0.018)).build()
    vector = lambda theta: simulation.run(theta).final.vector
    angle = lambda theta: simulation.run(theta).final.theta
    for medium in (1.2, 1.8):
        point = xp.array([medium])
        expected = [0.0012+0.02*0.018/medium, 0.018/medium]
        np.testing.assert_allclose(vector(point), expected)
        np.testing.assert_allclose(jit(vector)(point), expected)
        np.testing.assert_allclose(jit(jacobian(vector))(point), [[-0.02*0.018/medium**2], [-0.018/medium**2]])
        np.testing.assert_allclose(jit(hessian(angle))(point), [[2*0.018/medium**3]])


@pytest.mark.parametrize("radius", [0.2, -0.3, np.inf])
def test_mirror_is_unfolded_with_reflective_power_and_monotonic_z(radius):
    first, mirror, last = Space(d=0.1), Mirror(R=radius), Space(d=0.05)
    result = System(first, mirror, last, optical_input=Ray(x=0.001, theta=0.002, n=1)).build().run()
    height = 0.001 + 0.1 * 0.002
    angle = 0.002 - 2 * height / radius
    np.testing.assert_allclose(result[mirror].after.vector, [height, angle])
    np.testing.assert_allclose(result.final.vector, [height + 0.05 * angle, angle])
    np.testing.assert_allclose(result.z, [0, 0.1, 0.1, 0.15])
    assert result.final.n == 1


@pytest.mark.parametrize("composite, matrix, length", [
    (lambda: Telescope(f1=0.1, f2=0.2), [[-2.0, 0.3], [0.0, -0.5]], 0.3),
    (lambda: Telescope(f1=-0.05, f2=0.2), [[4.0, 0.15], [0.0, 0.25]], 0.15),
    (lambda: FourF(f1=0.1, f2=0.2), [[-2.0, 0.0], [0.0, -0.5]], 0.6),
], ids=["keplerian", "galilean", "four-f"])
def test_relay_composites_trace_their_independent_matrices(composite, matrix, length):
    element = composite()
    source = np.array([0.001, 0.01])
    result = System(element, optical_input=Ray(x=source[0], theta=source[1], n=1)).build().run()
    np.testing.assert_allclose(result.final.vector, np.array(matrix) @ source, atol=1e-15)
    assert result.z[-1] == pytest.approx(length)
    assert result[element.leaf_elements[0]].before is result.initial


@pytest.mark.parametrize("composite", [lambda: Telescope(f1=-0.2, f2=0.1), lambda: FourF(f1=-0.2, f2=0.1)],
                         ids=["telescope", "four-f"])
def test_relay_with_negative_total_spacing_is_infeasible_at_build(composite):
    with pytest.raises(ValueError, match="unsatisfiable"):
        System(composite(), optical_input=Ray(x=0.001, theta=0.01, n=1)).build()
