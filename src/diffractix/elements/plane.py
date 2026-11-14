"""
Defines the Plane element.
"""

from dataclasses import dataclass

from .element import OpticalElement


@dataclass(eq=False, kw_only=True)
class Plane(OpticalElement):
    """A reference plane: a named position with no effect on the light.

    Use it to read results at a position, or to place targets and constraints there.

    Parameters
    ----------
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Plane1"``.
    """

    @property
    def matrix(self):
        """Ray-transfer matrix: the identity."""
        return (
            (1.0, 0.0),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        """Element length [m]: 0."""
        return 0.0