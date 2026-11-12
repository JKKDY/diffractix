"""
Defines the dielectric Interface element.
"""

from dataclasses import dataclass

import numpy as np

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class Interface(OpticalElement):
    """A flat or curved boundary between two media, where light refracts.

    Parameters
    ----------
    n1 : float or Node
        Refractive index before the interface.
    n2 : float or Node
        Refractive index after the interface.
    R : float or Node, optional
        Radius of curvature [m]: positive when the centre of curvature lies after
        the interface (convex towards the incoming light), negative when it lies
        before. Defaults to ``inf``, a flat interface.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Interface1"``.

    Raises
    ------
    ValueError
        If `R` is zero when the system is built.

    Notes
    -----
    Requires ``n1 > 0`` and ``n2 > 0``. `n1` must match the medium the light
    arrives in, or the system cannot be built.
    """

    n1: Node
    n2: Node
    R: Node = np.inf

    def __post_init__(self):
        super().__post_init__()
        self.require(self.n1 > 0, self.n2 > 0)

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
            raise ValueError("Interface radius R must not be zero.")

    @property
    def matrix(self):
        # Keep a fixed infinite radius out of products with variable indices.
        """Ray-transfer matrix: ``((1, 0), ((n1 - n2) / (n2 * R), n1 / n2))``, for physical angles."""
        return (
            (1.0, 0.0),
            (((self.n1 - self.n2) / self.n2) / self.R, self.n1 / self.n2),
        )

    @property
    def element_length(self):
        """Element length [m]: 0."""
        return 0.0

    @property
    def refractive_index(self):
        """Refractive index: `n2`, the medium after the interface."""
        return self.n2
