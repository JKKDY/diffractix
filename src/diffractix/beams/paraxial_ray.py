from dataclasses import dataclass

import autograd.numpy as np


@dataclass
class ParaxialRay(ParaxialState):
    """
    Represents a paraxial ray at a specific point in space and medium.

    Attributes:
        x (float): Transverse ray position.
        theta (float): Ray angle relative to the optical axis in radians.
        n (float): The refractive index of the medium the ray is currently in.
    """
    x: float
    theta: float
    n: float

    def propagate(self, A, B, C, D, n) -> 'ParaxialRay':
        x = A * self.x + B * self.theta
        theta = C * self.x + D * self.theta

        return ParaxialRay(
            x=x,
            theta=theta,
            n=n,
        )

    @property
    def vector(self):
        """Ray vector [x, theta]."""
        return np.array([self.x, self.theta])


    def x_at_z(self, z: float) -> float:
        """Transverse position after free propagation over distance z."""
        return self.x + z * self.theta


    def __str__(self) -> str:
        """Human-readable representation of the ray state."""

        def fmt_length(val):
            if abs(val) < 1e-6: return f"{val*1e9:.1f} nm"
            if abs(val) < 1e-3: return f"{val*1e6:.1f} um"
            if abs(val) < 1.0: return f"{val*1e3:.1f} mm"
            if abs(val) >= 1e3: return f"{val/1e3:.1f} km"
            return f"{val:.2f} m"

        return (
            f"ParaxialRay("
            f"x={fmt_length(self.x)}, "
            f"theta={self.theta:.4g} rad, "
            f"n={self.n}"
            f")"
        )