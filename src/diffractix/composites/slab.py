"""
Defines the Slab composite element.
"""

from .composite import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node
from ..system import AMBIENT_N


class Slab(CompositeElement):
    """A block of material with flat faces, such as a window, filter or crystal.

    Parameters
    ----------
    d : float or Node
        Thickness [m].
    n : float or Node
        Refractive index of the material.
    ambient_n : float or Node, optional
        Refractive index outside the block. Defaults to the system's ambient index.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Slab1"``.

    Attributes
    ----------
    front : Interface
        Entry face, from `ambient_n` to `n`.
    body : Space
        The material, of length `d`.
    back : Interface
        Exit face, from `n` to `ambient_n`.
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