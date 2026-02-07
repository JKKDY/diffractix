"""
Defines the Gaussian Aperture element.
"""

from dataclasses import dataclass
from .element import OpticalElement
import autograd.numpy as np
from ..graph import Parameter, Constant, Symbol


@dataclass(kw_only=True)
class GaussianAperture(OpticalElement):
    """
    A 'soft' aperture with a Gaussian transmission profile: T(r) = exp(-r^2 / a^2).
    
    This element reduces the beam's spot size and intensity without changing 
    the radius of curvature, modeled using a complex ABCD matrix.

    Parameters:
        a (float): The radial width (1/e amplitude) of the aperture in meters.
        wavelength (float): The wavelength of the beam in meters.
    """

    a: float | Parameter
    wavelength: float | Parameter = Symbol("ambient_wavelength")

    @property
    def element_length(self):
        return Constant(0.0) # Apertures are assumed to have no physical thickness


    def compute_matrix(self, a, wavelength):
        """
        Returns the complex Gaussian aperture transformation matrix:
        [[ 1.0,           0.0 ],
         [ -i*λ / (π*a²), 1.0 ]]
        """
        # The C element is complex: -1j * wavelength / (pi * a^2)
        c_element = -1j * wavelength / (np.pi * a**2)
        
        return np.array([
            [1.0, 0.0],
            [c_element, 1.0]
        ], dtype=complex)