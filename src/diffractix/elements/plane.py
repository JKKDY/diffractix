"""
Defines the Plane element.
"""

from dataclasses import dataclass

from .element import OpticalElement


@dataclass(kw_only=True)
class Plane(OpticalElement):
    """
    A zero-length, zero-power reference plane.

    Plane does not modify the beam. It provides a stable named location in the
    optical system for inspection, targets, constraints, and result lookup.
    """

    @property
    def matrix(self):
        return (
            (1.0, 0.0),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return 0.0