"""
Defines the Slab composite element.
"""

from .sequence import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node
from ..core.system_vars import AMBIENT_N


class Slab(CompositeElement):
    """
    A sequence representing a physical block of material (Window, Filter, Crystal).
    """

    d: Node
    n: Node
    n_ambient: Node

    def __init__(self, d: float, n: float, n_ambient: float | Node = AMBIENT_N, label: str = "Slab"):
        self.d = d
        self.n = n
        self.n_ambient = n_ambient
        self.label = label

        self.front = Interface(n1=self.n_ambient, n2=self.n, label=f"{label}_In")
        self.body = Space(d=self.d, n=self.n, label=f"{label}_Body")
        self.back = Interface(n1=self.n, n2=self.n_ambient, label=f"{label}_Out")

        super().__init__()