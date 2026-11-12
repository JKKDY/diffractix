"""
Defines the Gradient-Index (GRIN) medium element.
"""

from dataclasses import dataclass

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class GRIN(OpticalElement):
    """A gradient-index (GRIN) medium with a parabolic index profile.

    The index falls off with distance ``r`` from the axis as
    ``n(r) = n * (1 - (g * r)**2 / 2)``. Rays follow sine-shaped paths about the
    axis with period ``2 * pi / g``, so the medium acts like a sequence of lenses.

    Parameters
    ----------
    d : float or Node
        Length [m].
    g : float or Node
        Gradient constant [1/m].
    n : float or Node
        Refractive index on the axis.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"GRIN1"``.

    Raises
    ------
    ValueError
        If `d` is a negative number.

    Notes
    -----
    Requires ``d >= 0`` and ``n > 0``. To enter or leave the medium from a
    different index, place an `Interface` on each side.
    """

    d: Node
    g: Node
    n: Node

    def __post_init__(self):
        super().__post_init__()
        
        length = self.element_length
        if isinstance(length, Node):
            self.require(length >= 0)
        elif length < 0:
            raise ValueError("Element length cannot be negative.")
        self.require(self.n > 0)

    @property
    def matrix(self):
        """Ray-transfer matrix: ``((cos(g d), sin(g d) / g), (-g sin(g d), cos(g d)))``."""
        phase = self.g * self.d
        return (
            (phase.cos(), self.d * phase.sinc()),
            (-self.g * phase.sin(), phase.cos()),
        )

    @property
    def element_length(self):
        """Element length [m]: `d`."""
        return self.d

    @property
    def refractive_index(self):
        """Refractive index: `n`, the on-axis index inside the medium."""
        return self.n
