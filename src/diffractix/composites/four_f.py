"""
Defines the 4f System composite element.
"""

from .composite import CompositeElement
from .telescope import Telescope
from ..elements.space import Space
from ..graph import Node


class FourF(CompositeElement):
    """A 4f relay: a telescope with one focal length of space before and after it.

    The input plane is imaged onto the output plane with magnification
    ``-f2 / f1``, and its Fourier transform lies inside the telescope, ``f1`` after
    the first lens.

    Parameters
    ----------
    f1 : float or Node
        Focal length of the first lens [m].
    f2 : float or Node
        Focal length of the second lens [m].
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"FourF1"``.

    Attributes
    ----------
    input_space : Space
        Gap of length `f1` before the telescope.
    telescope : Telescope
        The two lenses, ``f1 + f2`` apart: ``telescope.lens1``, ``telescope.space``
        and ``telescope.lens2``.
    output_space : Space
        Gap of length `f2` after the telescope.
    """

    f1: Node
    f2: Node

    def __init__(self, f1: float, f2: float, label: str | None = None):
        super().__init__(label)
        self.f1 = f1
        self.f2 = f2

        self.input_space = Space(d=self.f1, label=f"{self.label}_In_Drift")
        self.telescope = Telescope(f1=self.f1, f2=self.f2, label=f"{self.label}_Telescope")
        self.output_space = Space(d=self.f2, label=f"{self.label}_Out_Drift")
