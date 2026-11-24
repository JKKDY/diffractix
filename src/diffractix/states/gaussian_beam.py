from dataclasses import dataclass
from numbers import Real
from typing import ClassVar
import numpy as np
from diffractix.backend import xp, require

from .paraxial_state import ParaxialState
from diffractix.graph import Node
from diffractix.system.symbols import AMBIENT_N, WAVELENGTH, make_symbol


# Use backend.require() for (jit) tracer-safe validation.
# Concrete invalid values raise; dynamically traced checks are skipped.
def _require_real(value, name):
    """Inspect static numerical types without converting differentiable values."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        real = isinstance(value, Real) and not isinstance(value, bool)
    else:
        real = xp.issubdtype(dtype, xp.integer) or xp.issubdtype(dtype, xp.floating)
    if not real:
        raise TypeError(f"{name} must be real-valued numerical input.")


def _require_positive(**values):
    for name, value in values.items():
        _require_real(value, name)
        require(xp.all(xp.isfinite(value) & (value > 0)), f"{name} must be finite and positive.")


def _require_q(q):
    require(
        xp.all(xp.isfinite(q) & (xp.imag(q) > 0)),
        "Invalid q-parameter: q must be finite with a positive imaginary part.",
    )



def _curvature(inv_q_real):
    plane = xp.abs(inv_q_real) < 1e-16
    # Avoid differentiating a division by zero in the inactive branch.
    return xp.where(plane, np.inf, 1.0 / xp.where(plane, 1.0, inv_q_real))


@dataclass
class GaussianBeam(ParaxialState):
    """A fundamental (TEM00) Gaussian beam at one position on the optical axis.

    The beam is canonically described by its complex beam parameter ``q``. It is
    usually created with a factory rather than from ``q`` directly:
    :meth:`from_waist`, :meth:`from_w_and_R`, :meth:`from_divergence` or
    :meth:`from_fiber_tip`.

    Parameters
    ----------
    q : complex
        Complex beam parameter [m]: ``q = z + i z_R``, where ``z`` is the distance
        past the waist and ``z_R`` the Rayleigh range. Its imaginary part must be
        positive.
    wavelength : float or Node
        Vacuum wavelength [m].
    n : float or Node, optional
        Refractive index of the medium. Defaults to the system's ambient index
        (``System(ambient_n=...)``).
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"GaussianBeam1"``.

    Raises
    ------
    ValueError
        If `q` is not finite or its imaginary part is not positive. Checked only
        for concrete values, not inside JIT-compiled code.

    See Also
    --------
    Ray, `RayBundle` : Geometric rays.

    Notes
    -----
    Width and curvature follow from ``1/q = 1/R - i * wavelength / (pi * n * w**2)``.
    A transformation with ray-transfer matrix ``[[A, B], [C, D]]`` maps ``q`` to
    ``(A q + B) / (C q + D)``.

    Signs: ``R > 0`` for a diverging beam and ``R < 0`` for a converging one;
    ``z0 > 0`` when the waist lies ahead.

    Examples
    --------
    >>> from diffractix import GaussianBeam, units
    >>> from diffractix.units import mm, mrad, nm
    >>> beam = GaussianBeam.from_waist(w0=0.5 * mm, wavelength=1064 * nm)
    >>> units.format(beam.numeric().divergence_angle, mrad)
    '0.6774 mrad'
    """
    q: complex
    wavelength: Node
    n: Node = make_symbol(AMBIENT_N)

    result_columns: ClassVar[tuple[str, ...]] = (
        "w",
        "R",
        "gouy_phase",
    )

    def __post_init__(self):
        super().__post_init__()
        _require_q(self.q)
        if not getattr(type(self), "_is_numerical", False):
            self.require(self.wavelength > 0, label=f"{self.label}: wavelength > 0")
            if self.n.node is not None:
                self.require(self.n > 0, label=f"{self.label}: n > 0")

    def propagate(self, A, B, C, D, n) -> 'GaussianBeam':
        """Return the beam after a first-order (ABCD) transformation.

        Parameters
        ----------
        A, B, C, D : float
            Elements of the ray-transfer matrix.
        n : float
            Refractive index after the transformation.

        Returns
        -------
        GaussianBeam
            The new beam, with ``q' = (A q + B) / (C q + D)``; ``self`` is unchanged.
        """
        q = (A * self.q + B) / (C * self.q + D)

        return type(self)(
            q=q,
            wavelength=self.wavelength,
            n=n,
        )


    #-----------
    # PROPERTIES
    #-----------
    @property
    def refractive_index(self):
        """Refractive index: the medium the beam is in (the same as ``n``)."""
        return self.n

    @property
    def bindings(self):
        """Bindings: values this beam provides to the system, namely its wavelength for elements that need it."""
        return { WAVELENGTH.key: self.wavelength }

    @property
    def w(self) -> float:
        """Beam radius [m]: where the intensity falls to 1/e² of its peak."""
        inv_q_imag = xp.imag(1.0 / self.q)
        return xp.sqrt(-self.wavelength / (np.pi * self.n * inv_q_imag))

    @property
    def R(self) -> float:
        """Wavefront radius of curvature [m]: positive when diverging, negative when converging, ``inf`` at the waist."""
        inv_q_real = xp.real(1.0 / self.q)
        return _curvature(inv_q_real)

    @property
    def z_r(self) -> float:
        """Rayleigh range [m]: the distance from the waist over which the area doubles."""
        # q = z + i*z_r
        return xp.imag(self.q)

    @property
    def z0(self) -> float:
        """Waist position [m]: distance to the waist, positive ahead (converging), negative behind (diverging)."""
        # q = (z - z_waist) + i*z_r
        # Real(q) = z - z_waist. If we define current z=0, then Real(q) = -z_waist.
        return -xp.real(self.q)

    @property
    def w0(self) -> float:
        """Waist radius [m]: the smallest radius the beam reaches."""
        # z_r = (pi * w0^2 * n) / lambda
        return xp.sqrt(self.z_r * self.wavelength / (np.pi * self.n))

    @property
    def divergence_angle(self) -> float:
        """Divergence [rad]: the far-field half-angle, ``wavelength / (pi * n * w0)``."""
        return self.wavelength / (np.pi * self.w0 * self.n)

    @property
    def NA(self) -> float:
        """Numerical aperture: ``n * divergence_angle``, for matching fibers or objectives."""
        # theta = lambda / (pi * w0 * n)
        # NA = n * theta = lambda / (pi * w0)
        return self.wavelength / (np.pi * self.w0)

    @property
    def gouy_phase(self) -> float:
        """Gouy phase [rad]: ``arctan(z / z_R)``, with ``z`` the distance past the waist."""
        # Re(q) is the distance z from the waist. Im(q) is z_r.
        return xp.arctan2(xp.real(self.q), xp.imag(self.q))

    @property
    def b(self) -> float:
        """Confocal parameter [m]: twice the Rayleigh range."""
        return 2 * self.z_r


    #--------------------
    # CALCULATION METHODS
    #--------------------

    def R_at_z(self, z: float) -> float:
        """Return the wavefront radius of curvature after free propagation.

        Parameters
        ----------
        z : float
            Distance [m] along the axis from the beam's position; negative values look back.

        Returns
        -------
        float
            Radius of curvature [m]. Assumes a homogeneous medium with no elements in between.
        """
        q_z = self.q + z
        _, R_new = self.w_R_from_q(q_z, self.wavelength, self.n) 
        return R_new

    def w_at_z(self, z: float) -> float:
        """Return the beam radius after free propagation.

        Parameters
        ----------
        z : float
            Distance [m] along the axis from the beam's position; negative values look back.

        Returns
        -------
        float
            Beam radius [m]. Assumes a homogeneous medium with no elements in between.
        """
        q_z = self.q + z
        w_new, _ = self.w_R_from_q(q_z, self.wavelength, self.n)
        return w_new

    def overlap_with(self, other: 'GaussianBeam') -> float:
        """Return the power coupling efficiency between this beam and another one.

        Both beams must be described at the same position, for example a beam
        arriving at a fiber and the fiber's mode.

        Parameters
        ----------
        other : GaussianBeam
            The beam to couple into, at the same position.

        Returns
        -------
        float
            Fraction of the power coupled, between 0 and 1:
            ``4 Im(1/q1) Im(1/q2) / |conj(1/q1) - 1/q2|**2``.

        Raises
        ------
        ValueError
            If the beams are in media of different index, or their wavelengths differ
            by more than a relative 1e-6.
        """
        require(
            xp.abs(self.n - other.n) <= 1e-6,
            "Cannot calculate overlap between beams in different media.",
        )

        require(
            xp.abs(self.wavelength - other.wavelength)
            <= 1e-6 * xp.maximum(xp.abs(self.wavelength), xp.abs(other.wavelength)),
            "Cannot calculate overlap between beams of different wavelengths.",
        )

        # Using the q-parameter mismatch formula for coupling efficiency
        q1 = self.q
        q2 = other.q

        # Mismatch factor
        # This is a generalized formula for fundamental mode overlap
        # eta = 4 / ( abs( (w1/w2) + (w2/w1) )^2 + ( pi*w1*w2/lambda * (1/R1 - 1/R2) )^2 )
        # But computing via complex q is faster:
        # Power Coupling = 4 / ( |(w1/w2) + (w2/w1)|^2 + ... )
        # A simpler complex form:
        # eta = 4 * Im(1/q1) * Im(1/q2) / | 1/q1* - 1/q2 |^2
        # Note: q1* is complex conjugate.
        inv_q1 = 1.0 / q1
        inv_q2 = 1.0 / q2
        inv_q1_imag = xp.imag(inv_q1)
        inv_q2_imag = xp.imag(inv_q2)

        numerator = 4 * inv_q1_imag * inv_q2_imag
        denominator = abs(xp.conj(inv_q1) - inv_q2)**2
        return numerator / denominator

    @staticmethod
    def w_R_from_q(q: complex, wavelength: float, n: float) -> tuple[float, float]:
        """Return the beam radius and radius of curvature for a complex beam parameter.

        Parameters
        ----------
        q : complex
            Complex beam parameter [m], with a positive imaginary part.
        wavelength : float
            Vacuum wavelength [m].
        n : float
            Refractive index of the medium.

        Returns
        -------
        w, R : float
            Beam radius [m] and radius of curvature [m] (``inf`` for a flat wavefront),
            from ``1/q = 1/R - i * wavelength / (pi * n * w**2)``.
        """
        _require_q(q)
        inv_q = 1.0 / q

        # need to use np.real/imag to ensure autograd compatibility
        inv_q_real = xp.real(inv_q)
        inv_q_imag = xp.imag(inv_q)

        # radius of curvature (real part of 1/q)
        R = _curvature(inv_q_real)

        # Beam waist 
        # Im(1/q) = - lambda / (pi * n * w^2)  =>  w = sqrt( -lambda / (pi * n * Im(1/q)) )        
        w_sq = -wavelength / (np.pi * n * inv_q_imag)
        w = xp.sqrt(w_sq)
        return w, R


    #----------------
    # FACTORY METHODS
    #----------------

    @classmethod
    def from_waist(cls, w0: float, wavelength: float, z0: float = 0.0, n: float = 1.0) -> 'GaussianBeam':
        """Create a beam from its waist radius and the waist position.

        Parameters
        ----------
        w0 : float
            Waist radius [m].
        wavelength : float
            Vacuum wavelength [m].
        z0 : float, optional
            Waist position [m] relative to where the beam starts: 0 (the default)
            starts the beam at its waist, a positive value puts the waist ahead, a
            negative one behind. The new beam reports it as `z0`.
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        GaussianBeam
            The new beam.

        Raises
        ------
        TypeError
            If an argument is not a real number.
        ValueError
            If `w0`, `wavelength` or `n` is not positive and finite, or `z0`
            is not finite.
        """
        _require_positive(w0=w0, wavelength=wavelength, n=n)
        _require_real(z0, "z0")
        require(xp.all(xp.isfinite(z0)), "z0 must be finite.")
        z_r = (np.pi * w0**2 * n) / wavelength

        # q = (z - z0) + i*z_r. At z=0 (current plane):
        q = -z0 + 1j * z_r

        return cls(q=q, wavelength=wavelength, n=n)

    @classmethod
    def from_w_and_R(cls, w: float, R: float, wavelength: float, n: float = 1.0) -> 'GaussianBeam':
        """Create a beam from its radius and wavefront curvature where it starts.

        Parameters
        ----------
        w : float
            Beam radius [m] where the beam starts.
        R : float
            Wavefront radius of curvature [m] where the beam starts: positive when
            diverging, negative when converging, ``inf`` at the waist.
        wavelength : float
            Vacuum wavelength [m].
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        GaussianBeam
            The new beam.

        Raises
        ------
        ValueError
            If `w`, `wavelength` or `n` is not positive and finite, or `R` is zero or NaN.
        """
        _require_positive(w=w, wavelength=wavelength, n=n)
        _require_real(R, "R")
        require(xp.all(~xp.isnan(R) & (R != 0)), "R must be nonzero and not NaN; infinity is allowed.")
        inv_q_imag = -(wavelength / (np.pi * n * w**2))
        plane = xp.isinf(R)
        inv_q_real = xp.where(plane, 0.0, 1.0 / xp.where(plane, 1.0, R))

        denom = inv_q_real + 1j * inv_q_imag
        q = 1.0 / denom

        return cls(q=q, wavelength=wavelength, n=n)

    @classmethod
    def from_divergence(cls, theta: float, wavelength: float, n: float = 1.0) -> 'GaussianBeam':
        """Create a beam at its waist from its far-field divergence.

        Parameters
        ----------
        theta : float
            Far-field divergence half-angle [rad].
        wavelength : float
            Vacuum wavelength [m].
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        GaussianBeam
            The new beam, at its waist, with ``w0 = wavelength / (pi * n * theta)``.

        Raises
        ------
        ValueError
            If an argument is not positive and finite.
        """
        _require_positive(theta=theta, wavelength=wavelength, n=n)
        # theta = lambda / (pi * w0 * n)  ->  w0 = lambda / (pi * theta * n)
        w0 = wavelength / (np.pi * theta * n)

        return cls.from_waist(w0, wavelength, z0=0.0, n=n)

    @classmethod
    def from_fiber_tip(cls, wavelength: float, NA: float = None, MFD: float = None, n: float = 1.0) -> 'GaussianBeam':
        """Create the beam leaving a single-mode fiber, at its waist on the fiber face.

        Give either the numerical aperture or the mode field diameter, not both.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength [m].
        NA : float, optional
            Numerical aperture of the mode (1/e² far-field): ``w0 = wavelength / (pi * NA)``.
        MFD : float, optional
            Mode field diameter [m]: ``w0 = MFD / 2``.
        n : float, optional
            Refractive index of the medium. Defaults to 1, not to the system's
            ambient index.

        Returns
        -------
        GaussianBeam
            The new beam.

        Raises
        ------
        ValueError
            If neither or both of `NA` and `MFD` are given, or a value is not
            positive and finite.
        """
        if NA is None and MFD is None:
            raise ValueError("Must provide either 'NA' or 'MFD' to characterize the fiber tip.")
        if NA is not None and MFD is not None:
            raise ValueError("Give either 'NA' or 'MFD', not both; they describe the same mode.")
        _require_positive(wavelength=wavelength, n=n)
        if NA is not None:
            _require_positive(NA=NA)
            # Physics: NA = n * sin(theta) approx n * theta. 
            # In fiber optics, NA is usually defined such that w0 = wavelength / (pi * NA)
            w0 = wavelength / (np.pi * NA)

        else:
            _require_positive(MFD=MFD)
            w0 = MFD / 2.0

        return cls.from_waist(w0, wavelength, z0=0.0, n=n)


    #----------
    # DEBUGGING
    #----------

    def __str__(self) -> str:
        """Return a readable summary: wavelength, radius and curvature here, and the waist."""
        state = self._for_display()
        if state is None:
            return self._describe_fields()
        if state is not self:
            return str(state)
        # calculate parameters
        w_current = self.w_at_z(0)
        R_current = self.R_at_z(0)
        w_waist = self.w0
        z_waist = self.z0

        # formatting helper
        def fmt(val, unit='m'):
            if abs(val) < 1e-6: return f"{val*1e9:.1f} nm"
            if abs(val) < 1e-3: return f"{val*1e6:.1f} um"
            if abs(val) < 1.0:  return f"{val*1e3:.1f} mm"
            if abs(val) >= 1e3: return f"{val/1e3:.1f} km"
            return f"{val:.2f} {unit}"

        # Inf handling for curvature
        R_str = "Plane" if np.isinf(R_current) or abs(R_current) > 1e10 else fmt(R_current)

        return (
            f"GaussianBeam("
            f"lam={fmt(self.wavelength)}, "
            f"Current[w={fmt(w_current)}, R={R_str}], "
            f"Waist[w0={fmt(w_waist)} @ z={fmt(z_waist)}]"
            f")"
        )

    def plot(self, span: float = None, points: int = 200):
        """Plot this beam's envelope around its position and show it.

        For whole systems use ``result.plot()`` instead.

        Parameters
        ----------
        span : float, optional
            Length of axis to plot [m], centred on the beam's position. Defaults to
            four times the larger of the Rayleigh range and the distance to the waist.
        points : int, optional
            Number of samples. Defaults to 200.

        Raises
        ------
        ValueError
            If a field has no value yet, such as an `n` left to the system's
            ambient index; give it a value, or plot the result of a run.
        ImportError
            If Matplotlib cannot be imported.
        """
        from diffractix.plotting.gaussian_beam import _plot_gaussian_beam

        return _plot_gaussian_beam(self, span, points)
