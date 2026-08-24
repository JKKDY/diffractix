"""
Defines the Gaussian Aperture element.
"""

from dataclasses import dataclass

import autograd.numpy as np

from .base import OpticalElement
from ..graph import Node, SystemVar


@dataclass(kw_only=True)
class GaussianAperture(OpticalElement):
    """
    A soft Gaussian aperture.

    Parameters:
        a: Radial 1/e amplitude width in meters.
        wavelength: Optical wavelength in meters.
    """

    a: Node
    wavelength: Node = SystemVar("wavelength")

    @property
    def matrix(self):
        return (
            (1.0, 0.0),
            (
                -1j * self.wavelength / (np.pi * self.a**2),
                1.0,
            ),
        )

    @property
    def element_length(self):
        return 0.0