"""
Defines the Gaussian Aperture element.
"""

from dataclasses import dataclass

import numpy as np

from .element import OpticalElement
from diffractix.graph import Node
from diffractix.system.symbols import WAVELENGTH, make_symbol

@dataclass(eq=False, kw_only=True)
class GaussianAperture(OpticalElement):
    """A soft aperture with Gaussian amplitude transmission ``exp(-r**2 / a**2)``.

    It narrows a Gaussian beam. Use it with `GaussianBeam` inputs only; the
    transmitted power is not tracked.

    Parameters
    ----------
    a : float or Node
        Radius [m] at which the amplitude transmission falls to 1/e.
    wavelength : float or Node, optional
        Vacuum wavelength [m]. Defaults to the wavelength of the beam.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"GaussianAperture1"``.

    Notes
    -----
    Requires ``a > 0``, which matters when it is variable.
    """

    a: Node
    wavelength: Node = make_symbol(WAVELENGTH)

    def __post_init__(self):
        super().__post_init__()
        self.require(self.a > 0)

    @property
    def matrix(self):
        """Ray-transfer matrix in a medium of index 1: ``((1, 0), (-1j * wavelength / (pi * a**2), 1))``.

        In a simulation, the index of the surrounding medium is used instead of 1.
        """
        return self._matrix_in_medium(1.0)

    # todo: get rid of this function. We can for sure use some clever symbol binding trick. E.g. auto generated per element key
    def _matrix_in_medium(self, n):
        """Use the resolved local index with the physical-angle ABCD convention."""
        return (
            (1.0, 0.0),
            (-1j * self.wavelength / (np.pi * n * self.a**2), 1.0)
        )

    @property
    def element_length(self):
        """Element length [m]: 0."""
        return 0.0
