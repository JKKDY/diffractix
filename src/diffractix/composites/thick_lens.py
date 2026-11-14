"""
Defines the Thick Lens composite element.
"""

import numpy as np

from .composite import CompositeElement
from ..elements.space import Space
from ..elements.interface import Interface
from ..graph import Node
from ..system import AMBIENT_N


class ThickLens(CompositeElement):
    """A lens with thickness and two curved surfaces.

    Parameters
    ----------
    d : float or Node
        Centre thickness [m].
    n : float or Node
        Refractive index of the lens material.
    R1 : float or Node, optional
        Radius of curvature of the front surface [m]. Defaults to ``inf``, flat.
    R2 : float or Node, optional
        Radius of curvature of the back surface [m]. Defaults to ``inf``, flat.
    ambient_n : float or Node, optional
        Refractive index outside the lens. Defaults to the system's ambient index.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"ThickLens1"``.

    Attributes
    ----------
    front : Interface
        Front surface, radius `R1`.
    body : Space
        The lens material, of length `d`.
    back : Interface
        Back surface, radius `R2`.

    Notes
    -----
    A radius is positive when its centre of curvature lies after the surface, so a
    biconvex lens has ``R1 > 0`` and ``R2 < 0``.
    """

    d: Node
    n: Node
    R1: Node
    R2: Node
    ambient_n: Node

    def __init__(self, d: float, n: float, R1: float = np.inf, R2: float = np.inf,
                 ambient_n: float | Node = AMBIENT_N, label: str | None = None):
        super().__init__(label)
        self.d = d
        self.n = n
        self.R1 = R1
        self.R2 = R2
        self.ambient_n = ambient_n

        self.front = Interface(n1=self.ambient_n, n2=self.n, R=self.R1, label=f"{self.label}_front")
        self.body = Space(d=self.d, n=self.n, label=f"{self.label}_body")
        self.back = Interface(n1=self.n, n2=self.ambient_n, R=self.R2, label=f"{self.label}_back")
