"""
Defines the Thin Lens element.
"""

from dataclasses import dataclass

from .element import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class ThinLens(OpticalElement):
    """
    An idealized thin lens that changes beam divergence without adding
    physical length.

    Parameters:
        f: Focal length in meters. Positive values are converging and negative
           values are diverging.
    """

    f: Node

    @property
    def matrix(self):
        return (
            (1.0, 0.0),
            (-1.0 / self.f, 1.0),
        )

    @property
    def element_length(self):
        return 0.0