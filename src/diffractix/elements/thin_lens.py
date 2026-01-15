"""
Defines the Thin Lens element.
"""

from dataclasses import dataclass
from .base import OpticalElement
import autograd.numpy as np



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
    f: float

    @property
    def length_param_names(self):
        """
        Explicitly declares that the lens has no variable thickness (it is always 0).
        """
        return []

    def get_matrix(self, f):
        """
        Returns the thin lens transformation matrix:
        [[ 1.0,  0.0],
         [-1/f,  1.0]]
        """
        return np.array([[1.0, 0.0], [-1.0/f, 1.0]])

    def get_sim_data(self):
        """
        Provides the simulation engine with the matrix function, a constant 
        zero-length function, and the current focal length.
        """
        return (
            self.get_matrix, 
            lambda f: 0.0, 
            None,
            [float(self.f)]
        )