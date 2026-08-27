"""
Defines the free Space element.
"""

from dataclasses import dataclass

from .element import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class Space(OpticalElement):
    """
    A homogeneous propagation region.

    Parameters:
        d: Physical propagation distance in meters.
        n: Refractive index of the medium. None means inherit the current
           medium during system build.
    """

    d: Node
    n: Node | None = None

    @property
    def matrix(self):
        return (
            (1.0, self.d),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return self.d

    @property
    def element_refractive_index(self):
        return self.n