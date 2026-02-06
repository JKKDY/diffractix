"""
Defines the Thin Lens element.
"""

from dataclasses import dataclass
from .element import OpticalElement
import autograd.numpy as np
from ..graph import Parameter, Constant, Symbol



@dataclass(kw_only=True)
class ThinLens(OpticalElement):
    """
    An idealized thin lens that changes beam divergence without adding physical length.

    The ThinLens element applies a phase curvature to the beam based on its 
    focal length 'f'. It follows the paraxial approximation where the lens 
    thickness is assumed to be negligible compared to the focal length.

    Parameters:
        f (float): Focal length in meters. 
                   Positive for converging lenses, negative for diverging lenses.
    """
    f: float | Parameter

    @property
    def element_length(self):
        return Constant(0.0) # Thin lenses are idealized as having no physical thickness

    def compute_matrix(self, f):
        """
        Returns the thin lens transformation matrix:
        [[ 1.0,  0.0],
         [-1/f,  1.0]]
        """
        return np.array([[1.0, 0.0], [-1.0/f, 1.0]])


