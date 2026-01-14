"""
Defines the Mirror element.
"""
from dataclasses import dataclass
import autograd.numpy as np
from .base import OpticalElement


@dataclass(kw_only=True)
class Mirror(OpticalElement):
    """
    A curved or flat mirror that reflects the beam.

    In a sequential paraxial simulation, we 'unfold' the optical path. 
    This means the simulation coordinate z continues to increase, but the 
    mirror applies a phase transformation equivalent to a lens with f = R/2.

    Parameters:
        R (float): Radius of curvature in meters.
                   R > 0 is concave (converging).
                   R < 0 is convex (diverging).
                   R = np.inf is a flat mirror.
    """
    R: float = np.inf

    @property
    def length_param_names(self):
        """Mirrors are thin interfaces; they add no physical length."""
        return []

    def get_matrix(self, R):
        """
        Returns the reflection matrix.
        Mathematically identical to a thin lens with f = R/2.
        [[ 1,   0],
         [-2/R, 1]]
        """
        # Avoid division by infinity for flat mirrors
        power = -2.0 / R if not np.isinf(R) else 0.0
        return np.array([[1.0, 0.0], [power, 1.0]])

    def get_sim_data(self):
        return (
            self.get_matrix, 
            lambda R: 0.0, 
            [float(self.R)]
        )