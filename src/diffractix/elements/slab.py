"""
Defines the Slab factory function.
"""
import autograd.numpy as np
from .sequence import ElementSequence
from .space import Space
from .dielectric_interface import DielectricInterface

class Slab(ElementSequence):
    """
    A sequence representing a physical block of material (Window, Filter, Crystal).
    """
    def __init__(self, d: float, n: float, n_ambient: float = 1.0, label: str = "Slab"):
        
        front = DielectricInterface(n1=n_ambient, n2=n, R=np.inf, label=f"{label}_In")
        body  = Space(d=d, n=n, label=f"{label}_Body")
        back  = DielectricInterface(n1=n, n2=n_ambient, R=np.inf, label=f"{label}_Out")

        elements = [front, body, back]

        aliases = {
            'd': [(1, 'd')],
            'n': [(0, 'n2'), (1, 'n'), (2, 'n1')]
        }
        
        super().__init__(elements, aliases)