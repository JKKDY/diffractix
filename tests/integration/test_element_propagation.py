"""Medium-aware optical mathematics after layout and graph compilation."""
import numpy as np
import pytest
from diffractix import GaussianAperture, GaussianBeam, GRIN, Interface, Parameter, Ray, Space, System
from diffractix.backend import grad,hessian,jit,xp


@pytest.mark.parametrize("index", [1.,1.5])
def test_gaussian_aperture_envelope_and_wavelength_binding(index):
    aperture=GaussianAperture(a=0.0006)
    result=System(aperture,optical_input=GaussianBeam(q=0.04j,wavelength=1e-6,n=index)).build().run()
    expected=1/(1/result.initial.q-1j*1e-6/(np.pi*index*0.0006**2))
    np.testing.assert_allclose(result.final.q,expected)
    np.testing.assert_allclose(1/result.final.w**2,1/result.initial.w**2+1/0.0006**2)
    assert np.isrealobj(result.final.wavelength)
    assert result.final.n==index


def test_repeated_aperture_uses_each_placements_local_medium():
    aperture=GaussianAperture(a=0.0006)
    interface=Interface(n1=1,n2=1.5)
    result=System(aperture,interface,aperture,optical_input=GaussianBeam(q=0.04j,wavelength=1e-6)).build().run()
    q1=1/(1/(0.04j)-1j*1e-6/(np.pi*0.0006**2))
    q2=1.5*q1
    expected=1/(1/q2-1j*1e-6/(np.pi*1.5*0.0006**2))
    np.testing.assert_allclose(result[aperture,0].after.q,q1)
    np.testing.assert_allclose(result.final.q,expected)


@pytest.mark.parametrize("gradient", [0.,2.5])
def test_grin_zero_limit_and_nonzero_matrix(gradient):
    length=0.03
    element=GRIN(d=length,g=gradient,n=1.4)
    result=System(element,optical_input=Ray(x=0.001,theta=0.02,n=1.4)).build().run()
    phase=gradient*length
    b=length if gradient==0 else np.sin(phase)/gradient
    expected=np.array([[np.cos(phase),b],[-gradient*np.sin(phase),np.cos(phase)]])@np.array([0.001,0.02])
    np.testing.assert_allclose(result.final.vector,expected)
    assert result.z[-1]==pytest.approx(length)


def test_aperture_radius_derivatives_include_complex_matrix_without_complex_wavelength():
    radius=Parameter(0.0006).variable(bounds=(0.0004,0.001))
    simulation=System(GaussianAperture(a=radius),optical_input=GaussianBeam(q=0.04j,wavelength=1e-6)).build()
    before=simulation.run().initial.w
    width_squared=lambda theta:simulation.run(theta).final.w**2
    def reference(theta):
        return 1/(1/before**2+1/theta[0]**2)
    point=xp.array([0.0007])
    for transform in (lambda f:f,grad,hessian):
        np.testing.assert_allclose(transform(width_squared)(point),transform(reference)(point),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(jit(transform(width_squared))(point),transform(reference)(point),rtol=1e-10,atol=1e-12)
