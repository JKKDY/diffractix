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
    """Many paraxial rays traced together through the same system.

    Usually created with :meth:`parallel` or :meth:`converging`.

    Parameters
    ----------
    x : array_like
        Transverse position of each ray [m], one-dimensional.
    theta : array_like
        Angle of each ray to the optical axis [rad]: the physical angle in the
        rays' medium, not the reduced angle ``n * theta``. Same shape as `x`.
    n : float or Node, optional
        Refractive index of the medium. Defaults to the system's ambient index.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"RayBundle1"``.

    Raises
    ------
    ValueError
        If `x` and `theta` differ in shape or are not one-dimensional.

    See Also
    --------
    Ray : A single ray.

    Examples
    --------
    >>> from diffractix import RayBundle
    >>> from diffractix.units import mm
    >>> bundle = RayBundle.parallel(radius=2 * mm, count=5)
    >>> [round(float(x) / mm, 6) for x in bundle.x]
    [-2.0, -1.0, 0.0, 1.0, 2.0]
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
        """Create evenly spaced rays parallel to the axis.

        Parameters
        ----------
        radius : float
            Half-width of the bundle [m]: the rays span ``-radius`` to ``+radius``.
            Zero puts every ray on the axis.
        count : int
            Number of rays, at least 1.
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        RayBundle
            The new bundle.

        Raises
        ------
        TypeError
            If `count` is not an integer.
        ValueError
            If `count` is less than 1, or `radius` is negative or not finite.
        """
        cls._check_count(count)
        cls._check_radius(radius)
        x = xp.linspace(-radius, radius, count)
        theta = xp.zeros_like(x)
        return cls(x=x, theta=theta, n=n)

    @classmethod
    def converging(cls, radius, count, focal_distance, n=1.0):
        """Create evenly spaced rays aimed at a point on the axis.

        The ray at height ``x`` gets the angle ``-x / focal_distance``, so all rays
        cross the axis at `focal_distance` ahead of where the bundle starts.

        Parameters
        ----------
        radius : float
            Half-width of the bundle [m].
        count : int
            Number of rays, at least 1.
        focal_distance : float
            Distance to the crossing point [m]. Negative values give rays diverging
            from a point behind the start; ``inf`` gives parallel rays.
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        RayBundle
            The new bundle.

        Raises
        ------
        TypeError
            If `count` is not an integer.
        ValueError
            If `count` is less than 1, `radius` is negative or not finite, or
            `focal_distance` is zero or NaN.
        """
        cls._check_count(count)
        cls._check_radius(radius)
        cls._check_focal_distance(focal_distance)
        x = xp.linspace(-radius, radius, count)
        theta = -x / focal_distance
        return cls(x=x, theta=theta, n=n)

    @property
    def refractive_index(self):
        """Refractive index: the medium the rays are in (the same as ``n``)."""
        return self.n

    @property
    def size(self) -> int:
        """Size: the number of rays."""
        return len(self.x)

    @property
    def vectors(self):
        """Ray vectors: an array of shape ``(2, size)``, positions [m] then angles [rad]."""
        return xp.stack((self.x, self.theta), axis=0)

    @property
    def centroid(self):
        """Centroid [m]: the mean ray position. NaN for an empty bundle."""
        return xp.mean(self.x) if self.size else math.nan

    @property
    def mean_angle(self):
        """Mean angle [rad]: the mean ray angle. NaN for an empty bundle."""
        return xp.mean(self.theta) if self.size else math.nan

    @property
    def position_spread(self):
        """Position spread [m]: the standard deviation of the positions. NaN for an empty bundle."""
        return xp.std(self.x) if self.size else math.nan

    @property
    def angular_spread(self):
        """Angular spread [rad]: the standard deviation of the angles. NaN for an empty bundle."""
        return xp.std(self.theta) if self.size else math.nan

    @property
    def rms_x(self):
        """RMS height [m]: the root-mean-square distance from the axis. NaN for an empty bundle."""
        return xp.sqrt(xp.mean(xp.square(self.x))) if self.size else math.nan


    def propagate(self, A, B, C, D, n) -> 'RayBundle':
        """Return the bundle after a first-order (ABCD) transformation.

        Parameters
        ----------
        A, B, C, D : float
            Elements of the ray-transfer matrix, acting on each ``[x, theta]``.
        n : float
            Refractive index after the transformation.

        Returns
        -------
        RayBundle
            The new bundle, of the same type; ``self`` is unchanged.
        """
        return type(self)(
            x=A * self.x + B * self.theta,
            theta=C * self.x + D * self.theta,
            n=n,
        )

    def x_at_z(self, z: float):
        """Return the transverse positions after free propagation.

        Parameters
        ----------
        z : float or array_like
            Distance [m] along the axis from the bundle's position; negative values
            look back. An array of shape ``(k, 1)`` gives positions at ``k`` distances.

        Returns
        -------
        ndarray
            Transverse positions [m]: ``x + z * theta`` per ray. ``theta`` is the
            physical angle, so the index does not enter. Assumes a homogeneous medium
            with no elements in between.
        """
        return self.x + z * self.theta



    def __str__(self) -> str:
        """Return a readable summary of the bundle."""
        return f"RayBundle(size={self.size}, n={self.n:.4g})"
