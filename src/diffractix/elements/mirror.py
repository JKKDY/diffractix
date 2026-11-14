"""
Defines the Mirror element.
"""

from dataclasses import dataclass

import numpy as np

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class Mirror(OpticalElement):
    """A flat or curved mirror.

    The optical path is unfolded: after the mirror, light keeps travelling in +z,
    so a reflecting system is laid out as a straight line.

    Parameters
    ----------
    R : float or Node, optional
        Radius of curvature [m]: positive for a concave (focusing) mirror,
        negative for a convex one. Defaults to ``inf``, a flat mirror. The focal
        length is ``R / 2``.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Mirror1"``.

    Raises
    ------
    ValueError
        If `R` is zero when the system is built.
    """

    R: Node = np.inf

    def validate(self):
        """Check the element, and that the radius is not zero.

        Raises
        ------
        ValueError
            If `R` is zero.
        """
        super().validate()
        try:
            radius = self.R.value
        except (AttributeError, RuntimeError):
            return
        if radius == 0:
            raise ValueError("Mirror radius R must not be zero.")

    @property
    def matrix(self):
        """Ray-transfer matrix: ``((1, 0), (-2/R, 1))``."""
        return (
            (1.0, 0.0),
            (-2.0 / self.R, 1.0),
        )

    @property
    def element_length(self):
        """Element length [m]: 0."""
        return 0.0
