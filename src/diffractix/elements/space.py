"""
Defines the free Space element.
"""

from dataclasses import dataclass
from .element import OpticalElement
import autograd.numpy as np
from ..graph import Parameter

@dataclass(kw_only=True)
class Space(OpticalElement):
    """
    A homogeneous medium through which a beam propagates.

    The Space element represents a physical gap (e.g., air, glass, or vacuum).
    Its ABCD matrix affects the beam's transverse position based on its 
    divergence, scaled by the refractive index.

    Parameters:
        d (float): Physical distance of the propagation.
        n (float): Refractive index of the medium (default: 1.0).
    """
    d: float | Parameter
    n: float | Parameter = 1.0

    @property
    def length_param_names(self):
        """
        Explicitly links length to 'd'. 
        Note that 'n' affects the optical path length, but not the physical length.
        """
        return ['d'] 

    def get_matrix(self, d, n):
        """
        Returns the standard translation matrix:
        [[1, d/n],
         [0, 1  ]]
        """
        return np.array([[1.0, d/n], [0.0, 1.0]])

    def get_sim_data(self):
        return (
            self.get_matrix,                # matrix function
            lambda d, n: d,                 # length function
            lambda d, n: n,                 # refractive index function
            [self.d, self.n]  # values: [d, n]
        )
