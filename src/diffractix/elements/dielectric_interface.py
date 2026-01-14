"""
Defines the DielectricInterface element.
"""
from dataclasses import dataclass
import autograd.numpy as np
from .base import OpticalElement

@dataclass(kw_only=True)
class DielectricInterface(OpticalElement):
    """
    A distinct boundary between two media with different refractive indices.

    This element applies Snell's Law (in the paraxial limit) to the beam.
    It changes the beam angle but not its position.

    .. critical::
        This element is **mandatory** whenever the refractive index changes 
        between two `Space` elements.
        
        If you have `Space(n=1.0)` followed by `Space(n=1.5)`, you **must** insert a `DielectricInterface(n1=1.0, n2=1.5)` between them. 
        Omitting this violates Snell's Law and renders the simulation physically invalid.

    Parameters:
        n1 (float): Refractive index of the incoming medium.
        n2 (float): Refractive index of the outgoing medium.
        R (float): Radius of curvature of the interface (default: inf/flat).
                   R > 0 means the center of curvature is in the +z direction (Convex).
    """
    n1: float
    n2: float
    R: float = np.inf

    @property
    def length_param_names(self):
        return []

    def get_matrix(self, n1, n2, R):
        """
        Paraxial Snell's Law Matrix:
        [[ 1,           0      ],
         [ (n1-n2)/(R*n2), n1/n2 ]]
        """
        if np.isinf(R):
            power_term = 0.0
        else:
            power_term = (n1 - n2) / (R * n2)
            
        return np.array([
            [1.0, 0.0],
            [power_term, n1 / n2]
        ])

    def get_sim_data(self):
        return (
            self.get_matrix, 
            lambda n1, n2, R: 0.0, 
            [float(self.n1), float(self.n2), float(self.R)]
        )