from dataclasses import dataclass

import autograd.numpy as np

from .paraxial_state import ParaxialState


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
    n: float



    @property
    def size(self) -> int:
        """Number of rays in the bundle."""
        return len(self.x)

    @property
    def vectors(self):
        """Ray vectors with shape (2, N)."""
        return np.stack((self.x, self.theta), axis=0)



    def propagate(self, A, B, C, D, n) -> 'RayBundle':
        """Propagate all rays through an ABCD matrix."""
        x = A * self.x + B * self.theta
        theta = C * self.x + D * self.theta

        return RayBundle(
            x=x,
            theta=theta,
            n=n,
        )

    def x_at_z(self, z: float):
        """Transverse ray positions after free propagation over distance z."""
        return self.x + z * self.theta



    def __str__(self) -> str:
        """Human-readable representation of the ray bundle state."""
        return (
            f"RayBundle("
            f"size={self.size}, "
            f"n={self.n}"
            f")"
        )