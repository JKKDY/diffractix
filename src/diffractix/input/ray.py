from dataclasses import dataclass
from typing import ClassVar

from diffractix.backend import xp
from ..graph import Node
from ..system.symbols import AMBIENT_N, make_symbol
from .paraxial_input import ParaxialState


@dataclass
class Ray(ParaxialState):
    """A single paraxial ray: its height and angle at one position on the optical axis.

    Parameters
    ----------
    x : float or Node
        Transverse position [m], measured from the optical axis.
    theta : float or Node
        Angle to the optical axis [rad]: the physical angle in the ray's medium,
        not the reduced angle ``n * theta`` used in some texts.
    n : float or Node, optional
        Refractive index of the medium. Defaults to the system's ambient index
        (``System(ambient_n=...)``).
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Ray1"``.

    See Also
    --------
    RayBundle : Many rays traced together.
    GaussianBeam : A beam with a finite width and diffraction.

    Examples
    --------
    >>> from diffractix import Ray, Space, System
    >>> from diffractix.units import mm, mrad
    >>> ray = Ray(x=1 * mm, theta=2 * mrad)
    >>> result = System(Space(d=100 * mm), optical_input=ray).build().run()
    >>> round(float(result.final.x) / mm, 6)
    1.2
    """

    x: Node
    theta: Node
    n: Node = make_symbol(AMBIENT_N)

    result_columns: ClassVar[tuple[str, ...]] = (
        "x",
        "theta",
    )

    def propagate(self, A, B, C, D, n) -> "Ray":
        """Return the ray after a first-order (ABCD) transformation.

        Parameters
        ----------
        A, B, C, D : float
            Elements of the ray-transfer matrix, acting on ``[x, theta]``.
        n : float
            Refractive index after the transformation.

        Returns
        -------
        Ray
            The new ray, of the same type; ``self`` is unchanged.
        """
        x = A * self.x + B * self.theta
        theta = C * self.x + D * self.theta

        return type(self)(
            x=x,
            theta=theta,
            n=n,
        )

    @property
    def refractive_index(self):
        """Refractive index: the medium the ray is in (the same as ``n``)."""
        return self.n

    @property
    def vector(self):
        """Ray vector: the array ``[x, theta]``, in [m] and [rad]."""
        return xp.array([self.x, self.theta])

    def x_at_z(self, z: float):
        """Return the transverse position after free propagation.

        Parameters
        ----------
        z : float
            Distance [m] along the axis from the ray's position; negative values look back.

        Returns
        -------
        float
            Transverse position [m]: ``x + z * theta``. ``theta`` is the physical
            angle, so the index does not enter. Assumes a homogeneous medium with no
            elements in between.
        """
        return self.x + z * self.theta

    def __str__(self) -> str:
        """Return a readable summary of the ray."""

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
