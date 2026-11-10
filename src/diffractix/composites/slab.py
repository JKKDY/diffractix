"""
Defines the Slab composite element.
"""

from .composite import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node
from ..system import AMBIENT_N


class Slab(CompositeElement):
    """
    A sequence representing a physical block of material (Window, Filter, Crystal).
    """

    d: Node
    n: Node
    ambient_n: Node

    def __init__(self, d: float, n: float, ambient_n: float | Node = AMBIENT_N, label: str | None = None):
        super().__init__(label)
        self.d = d
        self.n = n
        self.ambient_n = ambient_n

        self.front = Interface(n1=self.ambient_n, n2=self.n, label=f"{self.label}_In")
        self.body = Space(d=self.d, n=self.n, label=f"{self.label}_Body")
        self.back = Interface(n1=self.n, n2=self.ambient_n, label=f"{self.label}_Out")