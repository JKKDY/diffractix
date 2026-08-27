"""
Defines the 4f System composite element.
"""

from .sequence import CompositeElement
from ..elements.space import Space
from ..elements.thin_lens import ThinLens
from ..graph import Node


class FourF(CompositeElement):
    """
    A standard 4f optical relay/correlator system consisting of two lenses and appropriate spacing.

    Layout:
        Space(f1) -> Lens1(f1) -> Space(f1 + f2) -> Lens2(f2) -> Space(f2)

    This setup ensures that the input plane is Fourier transformed at the mid-plane
    and imaged (inverted) at the output plane.
    """

    f1: Node
    f2: Node

    def __init__(self, f1: float, f2: float, label: str = "4f_System"):
        self.f1 = f1
        self.f2 = f2
        self.label = label

        self.input_space = Space(d=self.f1, label=f"{label}_In_Drift")
        self.lens1 = ThinLens(f=self.f1, label=f"{label}_L1")
        self.fourier_space = Space(d=self.f1 + self.f2, label=f"{label}_Fourier_Drift")
        self.lens2 = ThinLens(f=self.f2, label=f"{label}_L2")
        self.output_space = Space(d=self.f2, label=f"{label}_Out_Drift")

        super().__init__()