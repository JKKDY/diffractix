"""
Defines the Thin Lens element.
"""

from dataclasses import dataclass

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class ThinLens(OpticalElement):
    """An ideal thin lens.

    Parameters
    ----------
    f : float or Node
        Focal length [m]: positive for a converging lens, negative for a diverging
        one. Must not be zero.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"ThinLens1"``.

    Raises
    ------
    ValueError
        If `f` is zero when the system is built.

    Examples
    --------
    A ray parallel to the axis crosses it one focal length behind the lens:

    >>> from diffractix import Ray, Space, System, ThinLens
    >>> from diffractix.units import mm
    >>> system = System(ThinLens(f=100 * mm), Space(d=100 * mm), optical_input=Ray(x=1 * mm, theta=0))
    >>> round(float(system.build().run().final.x) / mm, 9)
    0.0
    """

    f: Node

    def validate(self):
        """Check the element, and that the focal length is not zero.

        Raises
        ------
        ValueError
            If `f` is zero.
        """
        super().validate()
        try:
            focal_length = self.f.value
        except (AttributeError, RuntimeError):
            return
        if focal_length == 0:
            raise ValueError("ThinLens focal length f must not be zero.")

    @property
    def matrix(self):
        """Ray-transfer matrix: ``((1, 0), (-1/f, 1))``."""
        return (
            (1.0, 0.0),
            (-1.0 / self.f, 1.0),
        )

    @property
    def element_length(self):
        """Element length [m]: 0."""
        return 0.0
