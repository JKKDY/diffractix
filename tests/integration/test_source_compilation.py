"""Source fields and element fields share canonical theta and snapshot values."""
import numpy as np
import pytest
from diffractix import GaussianBeam, Parameter, Ray, RayBundle, Space, System
from diffractix.backend import grad,hessian,jit,xp


def test_source_and_optical_fields_share_one_parameter_and_derivatives():
    p=Parameter(0.02).variable()
    source=Ray(x=p,theta=0.01)
    simulation=System(Space(d=2*p),optical_input=source).build()
    assert len(simulation.parameter_data)==1
    function=lambda theta:simulation.run(theta).final.x
    point=xp.array([0.03])
    assert function(point)==pytest.approx(0.0306)
    np.testing.assert_allclose(jit(grad(function))(point),[1.02])
    np.testing.assert_allclose(jit(hessian(lambda t:function(t)**2))(point),[[2*1.02**2]])
    p.value=0.5
    source.theta=0.9
    assert simulation.run().final.x==pytest.approx(0.0204)


def test_source_array_views_are_owned_by_build_and_previous_results():
    original=np.array([[-0.001,0.002],[0.01,-0.02]])
    source=RayBundle(x=original[0],theta=original[1])
    simulation=System(Space(d=0.1),optical_input=source).build()
    first=simulation.run()
    expected=np.array(first.final.vectors)
    original[:]=99
    source.x=np.array([4.,5.])
    np.testing.assert_array_equal(first.final.vectors,expected)
    np.testing.assert_array_equal(simulation.run().final.vectors,expected)


def test_wavelength_snapshot_and_source_medium_are_consistent():
    wavelength=Parameter(1e-6).variable(bounds=(0.5e-6,2e-6))
    beam=GaussianBeam(q=0.02j,wavelength=wavelength,n=1.33)
    simulation=System(Space(d=0.03),optical_input=beam).build()
    first=simulation.run()
    wavelength.value=2e-6
    beam.q=9j
    np.testing.assert_allclose(simulation.run().final.q,0.03+0.02j)
    assert first.final.wavelength==pytest.approx(1e-6)
    updated=simulation.run([1.5e-6])
    assert updated.final.wavelength==pytest.approx(1.5e-6)
    assert updated.final.n==pytest.approx(1.33)
    assert updated.final.w/first.final.w==pytest.approx(np.sqrt(1.5))



@pytest.mark.parametrize("q", [-1j, 1., 0.])
def test_build_rejects_source_q_mutated_after_construction(q):
    source = GaussianBeam(q=1j, wavelength=1e-6, n=1)
    source.q = q
    with pytest.raises(ValueError, match="Invalid q-parameter"):
        System(Space(d=0.1), optical_input=source).build()
