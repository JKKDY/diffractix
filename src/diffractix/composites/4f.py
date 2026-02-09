"""
Defines the 4f System composite element.
"""

import autograd.numpy as np
from .sequence import ElementSequence
from ..elements.space import Space
from ..elements.thin_lens import ThinLens

class FourF(ElementSequence):
    """
    A standard 4f optical relay/correlator system consisting of two lenses and appropriate spacing.
    
    Layout:
        Space(f1) -> Lens1(f1) -> Space(f1 + f2) -> Lens2(f2) -> Space(f2)
        
    This setup ensures that the input plane is Fourier transformed at the mid-plane
    and imaged (inverted) at the output plane.
    """
    def __init__(self, f1: float, f2: float, label: str = "4f_System"):
        l1 = ThinLens(f=f1, label=f"{label}_L1")
        l2 = ThinLens(f=f2, label=f"{label}_L2")

        s_in = Space(d=l1.f1, label=f"{label}_In_Drift")
        s_mid = Space(d=l1.f1 + l2.f2, label=f"{label}_Fourier_Drift")
        s_out = Space(d=l2.f2, label=f"{label}_Out_Drift")
        
        elements = [s_in, l1, s_mid, l2, s_out]

        aliases = {
            'f1': [(0, 'd'), (1, 'f')],  # Maps to Input Space d and Lens1 f
            'f2': [(3, 'f'), (4, 'd')]   # Maps to Lens2 f and Output Space d
        }
        
        super().__init__(elements, aliases)