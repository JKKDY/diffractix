from dataclasses import dataclass
from typing import ClassVar

from diffractix.backend import xp
from .base import ParaxialState


@dataclass
class Ray(ParaxialState):
    """
    Represents a single paraxial ray at a specific point in space.

    Attributes:
        x: Transverse ray position in meters.
        theta: Ray angle relative to the optical axis in radians.
        n: Refractive index of the current medium.
    """

    x: float
    theta: float
    n: float = 1.0

    result_columns: ClassVar[tuple[str, ...]] = (
        "x",
        "theta",
    )

    def propagate(self, A, B, C, D, n) -> "Ray":
        """Propagate the ray through an ABCD optical system."""
        x = A * self.x + B * self.theta
        theta = C * self.x + D * self.theta

        return Ray(
            x=x,
            theta=theta,
            n=n,
        )

    @property
    def vector(self):
        """Ray vector [x, theta]."""
        return xp.array([self.x, self.theta])

    def x_at_z(self, z: float):
        """Transverse position after free propagation over distance z."""
        return self.x + z * self.theta

    def __str__(self) -> str:
        """Human-readable representation of the ray state."""

        def fmt_length(val):
            if abs(val) < 1e-6: return f"{val * 1e9:.1f} nm"
            if abs(val) < 1e-3: return f"{val * 1e6:.1f} um"
            if abs(val) < 1.0:  return f"{val * 1e3:.1f} mm"
            if abs(val) >= 1e3: return f"{val / 1e3:.1f} km"
            return f"{val:.2f} m"

        return (
            f"Ray("
            f"x={fmt_length(self.x)}, "
            f"theta={self.theta:.4g} rad, "
            f"n={self.n:.4g}"
            f")"
        )
