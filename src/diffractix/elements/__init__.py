# diffractix/elements/__init__.py

from .base import OpticalElement
from .space import Space
from .thin_lens import ThinLens
from .mirror import Mirror
from .dielectric_interface import DielectricInterface
from .abcd import ABCD

# Import factories
from .slab import Slab
from .thick_lens import ThickLens

__all__ = [
    "OpticalElement", 
    "Space", 
    "ThinLens", 
    "Mirror", 
    "DielectricInterface", 
    "ABCD",
    "Slab",
    "ThickLens"
]