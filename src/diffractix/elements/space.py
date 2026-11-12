"""
Defines the free Space element.
"""

from dataclasses import dataclass

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class Space(OpticalElement):
    """Free propagation through a homogeneous medium.

    Parameters
    ----------
    d : float or Node
        Length [m].
    n : float or Node, optional
        Refractive index of the medium inside the space. Defaults to ``None``:
        the medium before the space continues.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Space1"``.

    Raises
    ------
    ValueError
        If `d` is a negative number.

    Notes
    -----
    Requires ``d >= 0`` and ``n > 0``, which matters when they are variable.
    """

    d: Node
    n: Node | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.n.node is not None:
            self.require(self.n > 0)
        
        length = self.element_length
        if isinstance(length, Node):
            self.require(length >= 0)
        elif length < 0:
            raise ValueError("Element length cannot be negative.")

    @property
    def matrix(self):
        """Ray-transfer matrix: ``((1, d), (0, 1))``."""
        return (
            (1.0, self.d),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        """Element length [m]: `d`."""
        return self.d

    @property
    def refractive_index(self):
        """Refractive index: `n`, the medium inside the space, or ``None`` to continue the medium before it."""
        return self.n
