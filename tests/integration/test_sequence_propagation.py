"""Native Program output tuples retain numerical dtype semantics in simulation."""
import numpy as np
from diffractix import ABCD, GaussianBeam, Parameter, RayBundle, Space, System
from diffractix.backend import grad,hessian,jit,xp
from diffractix.graph import Literal


def test_complex_optical_roots_do_not_promote_real_source_fields():
    p=Parameter(0.02).variable()
    element=ABCD(A=1,B=p,C=Literal(1j*0.1),D=1,thickness=p)
    simulation=System(element,optical_input=GaussianBeam(q=0.04j,wavelength=1e-6)).build()
    function=lambda theta:simulation.run(theta).final.w**2
    def expected(theta):
        q=(0.04j+theta[0])/(1+0.1j*0.04j)
        return -1e-6/(np.pi*xp.imag(1/q))
    point=xp.array([0.03])
    result=simulation.run(point)
    assert np.isrealobj(result.initial.wavelength)
    assert np.isrealobj(result.final.n)
    for transform in (lambda f:f,grad,hessian):
        np.testing.assert_allclose(transform(function)(point),transform(expected)(point),rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(jit(transform(function))(point),transform(expected)(point),rtol=1e-11,atol=1e-12)


def test_float32_bundle_promotion_matches_explicit_scalar_propagation():
    positions=xp.array([-0.001,0.002],dtype=xp.float32)
    angles=xp.array([0.01,-0.02],dtype=xp.float32)
    simulation=System(Space(d=0.125),optical_input=RayBundle(x=positions,theta=angles)).build()
    result=simulation.run()
    # Native roots preserve Python scalar semantics without packed-array promotion.
    expected=positions+0.125*angles
    np.testing.assert_allclose(result.final.x,expected,rtol=1e-7,atol=1e-12)
    assert result.final.x.dtype==expected.dtype
    np.testing.assert_array_equal(result.initial.x,positions)
