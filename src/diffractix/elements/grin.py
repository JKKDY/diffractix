"""
Defines the Gradient-Index (GRIN) medium element.
"""

from dataclasses import dataclass

from .base import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class GRIN(OpticalElement):
    """
    A parabolic gradient-index medium.

    Parameters:
        d: Physical length of the medium in meters.
        g: Gradient constant in inverse meters.
        n: On-axis refractive index.
    """

    d: Node
    g: Node
    n: Node

    @property
    def matrix(self):
        phase = self.g * self.d
        return (
            (phase.cos(), phase.sin() / self.g),
            (-self.g * phase.sin(), phase.cos()),
        )

    @property
    def element_length(self):
        return self.d

    @property
    def element_refractive_index(self):
        return self.n