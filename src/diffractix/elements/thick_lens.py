"""
Defines the Thick Lens element.
"""


import autograd.numpy as np
from .sequence import ElementSequence
from .space import Space
from .dielectric_interface import DielectricInterface

class ThickLens(ElementSequence):
    """
    A physical lens with thickness d, index n, and surface radii R1, R2.
    """
    def __init__(self, d: float, n: float, R1: float = np.inf, R2: float = np.inf, 
                 n_ambient: float = 1.0, label: str = "ThickLens"):
        
        # A thick lens is modeled as two interfaces separated by a space
        front = DielectricInterface(n1=n_ambient, n2=n, R=R1, label=f"{label}_front")
        body  = Space(d=d, n=n, label=f"{label}_body")
        back  = DielectricInterface(n1=n, n2=n_ambient, R=R2, label=f"{label}_back")
        
        elements = [front, body, back]

        # Define Aliases 
        # e.g. when user says .variable('n'), we update it in all 3 elements places.
        aliases = {
            'R1': [(0, 'R')],
            'R2': [(2, 'R')],
            'd':  [(1, 'd')],
            'n':  [(0, 'n2'), (1, 'n'), (2, 'n1')]
        }

        super().__init__(elements, aliases)