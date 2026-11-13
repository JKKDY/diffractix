"""
Defines the Telescope composite element.
"""

from .composite import CompositeElement
from ..elements.space import Space
from ..elements.thin_lens import ThinLens
from ..graph import Node


class Telescope(CompositeElement):
    """Two thin lenses spaced ``f1 + f2`` apart: an afocal telescope.

    A collimated input leaves collimated, with its radius scaled by ``f2 / f1``.
    With two positive lenses it is a Keplerian telescope (the image is inverted);
    with a negative first lens, a Galilean one.

    Parameters
    ----------
    f1 : float or Node
        Focal length of the first lens [m].
    f2 : float or Node
        Focal length of the second lens [m].
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Telescope1"``.

    Attributes
    ----------
    lens1 : ThinLens
        First lens, focal length `f1`.
    space : Space
        Gap of length ``f1 + f2``.
    lens2 : ThinLens
        Second lens, focal length `f2`.

    Examples
    --------
    >>> from diffractix import Ray, System, Telescope
    >>> from diffractix.units import mm
    >>> system = System(Telescope(f1=50 * mm, f2=100 * mm), optical_input=Ray(x=1 * mm, theta=0))
    >>> round(float(system.build().run().final.x) / mm, 9)
    -2.0
    """

    f1: Node
    f2: Node

    def __init__(self, f1: float, f2: float, label: str | None = None):
        super().__init__(label)
        self.f1 = f1
        self.f2 = f2

        self.lens1 = ThinLens(f=self.f1, label=f"{self.label}_L1")
        self.space = Space(d=self.f1 + self.f2, label=f"{self.label}_Drift")
        self.lens2 = ThinLens(f=self.f2, label=f"{self.label}_L2")