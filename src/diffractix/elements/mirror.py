"""
Defines the Mirror element.
"""

from dataclasses import dataclass

import autograd.numpy as np

from .base import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class Mirror(OpticalElement):
    """
    A curved or flat mirror that reflects the beam.

    In a sequential paraxial simulation, the optical path is unfolded so that
    z continues to increase through the reflection.

    Parameters:
        R: Radius of curvature in meters.
           R > 0 is concave (converging).
           R < 0 is convex (diverging).
           R = inf is flat.
    """

    R: Node = np.inf

    @property
    def matrix(self):
        return (
            (1.0, 0.0),
            (-2.0 / self.R, 1.0),
        )

    @property
    def element_length(self):
        return 0.0