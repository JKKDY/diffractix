"""
Defines the free Space element.
"""

from dataclasses import dataclass
from .element import OpticalElement
import autograd.numpy as np
from ..graph import Parameter, Symbol

@dataclass(kw_only=True)
class Space(OpticalElement):
    """
    A homogeneous medium through which a beam propagates.

    The Space element represents a physical gap (e.g., air, glass, or vacuum).
    Its ABCD matrix affects the beam's transverse position based on its 
    divergence, scaled by the refractive index.

    Parameters:
        d (float): Physical length of the object (not the optical length!).
        n (float): Refractive index of the medium (default: ambient system index).
    """

    d: float | Parameter
    n: float | Parameter = None 

    @property
    def element_length(self):
        return self.d

    @property
    def element_refractive_index(self):
        return self.n

    def compute_matrix(self, d, n):
        """
        Returns the standard translation matrix for (y, theta) convention.
        
        Physics Note: 
        Even though the medium has index n, the B-element is simply 'd' (physical distance).
        The refractive index 'n' affects the beam via the q-parameter definition, 
        not the translation matrix itself in this convention.
        
        [[1, d],
         [0, 1]]
        """
        return np.array([[1.0, d], [0.0, 1.0]])

   