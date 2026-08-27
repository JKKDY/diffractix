"""
Defines the dielectric Interface element.
"""

from dataclasses import dataclass

import autograd.numpy as np

from .element import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class Interface(OpticalElement):
    """
    A boundary between two media with different refractive indices.

    Parameters:
        n1: Incoming refractive index.
        n2: Outgoing refractive index.
        R: Radius of curvature. Infinite radius represents a flat interface.
    """

    n1: Node
    n2: Node
    R: Node = np.inf

    @property
    def matrix(self):
        return (
            (1.0, 0.0),
            ((self.n1 - self.n2) / (self.R * self.n2), self.n1 / self.n2),
        )

    @property
    def element_length(self):
        return 0.0

    @property
    def element_refractive_index(self):
        return self.n2