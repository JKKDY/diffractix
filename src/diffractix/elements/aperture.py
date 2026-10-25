"""
Defines the Gaussian Aperture element.
"""

from dataclasses import dataclass, field

import numpy as np

from .element import OpticalElement
from diffractix.graph import Node, Symbol
from diffractix.system.symbols import WAVELENGTH, make_symbol

@dataclass(eq=False, kw_only=True)
class GaussianAperture(OpticalElement):
    """
    A soft Gaussian aperture.

    Parameters:
        a: Radial 1/e amplitude width in meters.
        wavelength: Optical wavelength in meters.
    """

    a: Node
    wavelength: Node = make_symbol(WAVELENGTH)

    def __post_init__(self):
        super().__post_init__()
        self.require(self.a > 0)

    @property
    def matrix(self):
        """Unit-index matrix; System compilation supplies the local medium."""
        return self._matrix_in_medium(1.0)

    def _matrix_in_medium(self, n):
        """Use the resolved local index with the physical-angle ABCD convention."""
        return (
            (1.0, 0.0),
            (-1j * self.wavelength / (np.pi * n * self.a**2), 1.0)
        )

    @property
    def element_length(self):
        return 0.0
