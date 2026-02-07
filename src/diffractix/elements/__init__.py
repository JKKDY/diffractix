# diffractix/elements/__init__.py

from .element import OpticalElement
from .space import Space
from .thin_lens import ThinLens
from .mirror import Mirror
from .interface import Interface
from .aperture import GaussianAperture
from .abcd import ABCD


__all__ = [
    "OpticalElement", 
    "Space", 
    "ThinLens", 
    "Mirror", 
    "Interface", 
    "ABCD",
    "GaussianAperture"
]