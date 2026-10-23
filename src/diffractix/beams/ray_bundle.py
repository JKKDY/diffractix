from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from diffractix.backend import xp

from .base import ParaxialState




@dataclass
class RayBundle(ParaxialState):
    """
    Represents a bundle of paraxial rays at a specific point in space and medium.

    Attributes:
        x (array): Transverse ray positions.
        theta (array): Ray angles relative to the optical axis in radians.
        n (float): The refractive index of the medium the rays are currently in.
    """
    x: np.ndarray
    theta: np.ndarray
    n: float = 1.0

    result_columns: ClassVar[tuple[str, ...]] = (
        "x",
        "theta",
    )

    def __post_init__(self):
        if np.shape(self.x) != np.shape(self.theta):
            raise ValueError("x and theta must have matching shapes.")
        if len(np.shape(self.x)) != 1:
            raise ValueError("RayBundle arrays must be one-dimensional.")


    @property
    def size(self) -> int:
        """Number of rays in the bundle."""
        return len(self.x)

    @property
    def vectors(self):
        """Ray vectors with shape (2, N)."""
        return xp.stack((self.x, self.theta), axis=0)



    def propagate(self, A, B, C, D, n) -> 'RayBundle':
        """Propagate all rays through an ABCD matrix."""
        return RayBundle(
            x=A * self.x + B * self.theta,
            theta=C * self.x + D * self.theta,
            n=n,
        )

    def x_at_z(self, z: float):
        """Transverse ray positions after free propagation over distance z."""
        return self.x + z * self.theta



    def __str__(self) -> str:
        """Human-readable representation of the ray bundle state."""
        return f"RayBundle(size={self.size}, n={self.n:.4g})"
