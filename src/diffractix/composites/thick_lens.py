"""
Defines the Thick Lens composite element.
"""

import autograd.numpy as np

from .composite import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node
from ..system import AMBIENT_N


class ThickLens(CompositeElement):
    """
    A physical lens with thickness d, index n, and surface radii R1, R2.
    """

    d: Node
    n: Node
    R1: Node
    R2: Node
    n_ambient: Node

    def __init__(self, d: float, n: float, R1: float = np.inf, R2: float = np.inf,
                 n_ambient: float | Node = AMBIENT_N, label: str = "ThickLens"):
        self.d = d
        self.n = n
        self.R1 = R1
        self.R2 = R2
        self.n_ambient = n_ambient
        self.label = label

        self.front = Interface(n1=self.n_ambient, n2=self.n, R=self.R1, label=f"{label}_front")
        self.body = Space(d=self.d, n=self.n, label=f"{label}_body")
        self.back = Interface(n1=self.n, n2=self.n_ambient, R=self.R2, label=f"{label}_back")

        super().__init__()