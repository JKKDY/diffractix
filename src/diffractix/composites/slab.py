"""
Defines the Slab composite element.
"""

from .sequence import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node


class Slab(CompositeElement):
    """
    A sequence representing a physical block of material (Window, Filter, Crystal).
    """

    d: Node
    n: Node

    def __init__(self, d: float, n: float, n_ambient: float = 1.0, label: str = "Slab"):
        self.d = d
        self.n = n
        self.label = label

        self.front = Interface(n1=n_ambient, n2=self.n, label=f"{label}_In")
        self.body = Space(d=self.d, n=self.n, label=f"{label}_Body")
        self.back = Interface(n1=self.n, n2=n_ambient, label=f"{label}_Out")

        super().__init__()