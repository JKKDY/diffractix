import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np
from diffractix.backend import xp
from ..graph import Node
from ..system.symbols import AMBIENT_N, make_symbol

from .paraxial_input import ParaxialState




@dataclass
class RayBundle(ParaxialState):
    """
    Represents a bundle of paraxial rays at a specific point in space and medium.

    Attributes:
        x (array): Transverse ray positions.
        theta (array): Ray angles relative to the optical axis in radians.
        n: Refractive-index graph input; defaults to the ambient-index Symbol.
    """
    x: np.ndarray
    theta: np.ndarray
    n: Node = make_symbol(AMBIENT_N)

    result_columns: ClassVar[tuple[str, ...]] = (
        "x",
        "theta",
    )

    def __post_init__(self):
        super().__post_init__()
        if np.shape(self.x) != np.shape(self.theta):
            raise ValueError("x and theta must have matching shapes.")
        if len(np.shape(self.x)) != 1:
            raise ValueError("RayBundle arrays must be one-dimensional.")
    
    @staticmethod
    def _check_count(count):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"count must be an integer, got {count!r}.")
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}.")

    @staticmethod
    def _check_radius(radius):
        # Only concrete numbers are checked; traced values pass through.
        if isinstance(radius, Real) and not (math.isfinite(radius) and radius >= 0):
            raise ValueError(f"radius must be finite and >= 0, got {radius}.")

    @staticmethod
    def _check_focal_distance(focal_distance):
        if isinstance(focal_distance, Real) and (focal_distance == 0 or math.isnan(focal_distance)):
            raise ValueError(f"focal_distance must be nonzero and not NaN, got {focal_distance}.")

    @classmethod
    def parallel(cls, radius, count, n=1.0):
        """Generate parallel rays across a 1D aperture of half-width radius."""
        cls._check_count(count)
        cls._check_radius(radius)
        x = xp.linspace(-radius, radius, count)
        theta = xp.zeros_like(x)
        return cls(x=x, theta=theta, n=n)

    @classmethod
    def converging(cls, radius, count, focal_distance, n=1.0):
        """Generate rays aimed at a point focal_distance ahead (negative: diverging; inf: parallel)."""
        cls._check_count(count)
        cls._check_radius(radius)
        cls._check_focal_distance(focal_distance)
        x = xp.linspace(-radius, radius, count)
        theta = -x / focal_distance
        return cls(x=x, theta=theta, n=n)

    @property
    def refractive_index(self):
        return self.n

    @property
    def size(self) -> int:
        """Number of rays in the bundle."""
        return len(self.x)

    @property
    def vectors(self):
        """Ray vectors with shape (2, N)."""
        return xp.stack((self.x, self.theta), axis=0)

    @property
    def centroid(self):
        """Mean transverse position of the rays (NaN for an empty bundle)."""
        return xp.mean(self.x) if self.size else math.nan

    @property
    def mean_angle(self):
        """Mean propagation angle (NaN for an empty bundle)."""
        return xp.mean(self.theta) if self.size else math.nan

    @property
    def position_spread(self):
        """RMS spread around the bundle centroid (NaN for an empty bundle)."""
        return xp.std(self.x) if self.size else math.nan

    @property
    def angular_spread(self):
        """RMS angular spread around the mean angle (NaN for an empty bundle)."""
        return xp.std(self.theta) if self.size else math.nan

    @property
    def rms_x(self):
        """RMS transverse displacement from the optical axis (NaN for an empty bundle)."""
        return xp.sqrt(xp.mean(xp.square(self.x))) if self.size else math.nan


    def propagate(self, A, B, C, D, n) -> 'RayBundle':
        """Propagate all rays through an ABCD matrix."""
        return type(self)(
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
