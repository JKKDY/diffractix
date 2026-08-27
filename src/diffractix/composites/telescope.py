"""
Defines the Telescope composite element.
"""

from .sequence import CompositeElement
from ..elements.space import Space
from ..elements.thin_lens import ThinLens
from ..graph import Node


class Telescope(CompositeElement):
    """
    A two-lens afocal telescope.

    Layout:
        Lens1(f1) -> Space(f1 + f2) -> Lens2(f2)

    For two positive focal lengths this represents a Keplerian telescope.
    """

    f1: Node
    f2: Node

    def __init__(self, f1: float, f2: float, label: str = "Telescope"):
        self.f1 = f1
        self.f2 = f2
        self.label = label

        self.lens1 = ThinLens(f=self.f1, label=f"{label}_L1")
        self.space = Space(d=self.f1 + self.f2, label=f"{label}_Drift")
        self.lens2 = ThinLens(f=self.f2, label=f"{label}_L2")

        super().__init__()