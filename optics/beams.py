
from dataclasses import dataclass
from typing import Tuple
import autograd.numpy as np
import cmath
from .core import Entity


@dataclass
class GaussianBeam(Entity):
    q: complex
    wavelength: float

    @property
    def z_r(self) -> float:
        """Rayleigh range (in terms of propagation distance)."""
        return self.q.imag

    @property
    def z0(self) -> float:
        """
        Position of the waist relative to current plane.
        If z0 is positive, the waist is in front of us (converging beam).
        If z0 is negative, the waist is behind us (diverging beam).
        """
        # q = z - z_waist + i*z_r. 
        # At current plane z=0, q = -z_waist + i*z_r.
        return -self.q.real

    def w0(self, n: float = 1.0) -> float:
        """Calculate waist radius w0 for a given refractive index n."""
        # z_r = (pi * w0^2 * n) / lambda
        # w0 = sqrt(z_r * lambda / (pi * n))
        return np.sqrt(self.z_r * self.wavelength / (np.pi * n))

    def w_at_z(self, z: float, n: float = 1.0) -> float:
        """Calculate spot size w at distance z from current plane."""
        # Calculate q at distance z
        q_z = self.q + z
        
        # Extract R and w from 1/q(z)
        # 1/q = 1/R - i * (lambda / (pi * n * w^2))
        inv_q = 1.0 / q_z
        imag_part = -inv_q.imag # This is lambda / (pi * n * w^2)
        
        return np.sqrt(self.wavelength / (np.pi * n * imag_part))

    def divergence_angle(self, n: float = 1.0) -> float:
        """Far-field divergence half-angle (radians)."""
        return self.wavelength / (np.pi * self.w0(n) * n)

    def curvature_at_z(self, z: float) -> float:
        """Radius of curvature R at distance z."""
        q_z = self.q + z
        if q_z.real == 0:
            return float('inf') # Plane wave at waist
        # R(z) = z_rel * (1 + (z_r/z_rel)^2)
        # Using 1/q formulation: 1/q = 1/R - ... => R = 1/real(1/q)
        inv_q = 1.0 / q_z
        return 1.0 / inv_q.real


    # --- Factory Methods ---
    @classmethod
    def from_waist(cls, w0: float, wavelength: float, z_waist_loc: float = 0.0, n: float = 1.0) -> 'GaussianBeam':
        """
        Create a beam defined by its waist size.
        z_waist_loc: Where the waist is relative to the *current* plane.
                     0.0 means we are at the waist.
                     -10.0 means the waist was 10 units ago.
        """
        z_r = (np.pi * w0**2 * n) / wavelength
        # q = (z - z_waist_loc) + i*z_r. At z=0 (current plane):
        q = -z_waist_loc + 1j * z_r
        return cls(q=q, wavelength=wavelength)

    @classmethod
    def from_w_and_R(cls, w: float, R: float, wavelength: float, n: float = 1.0) -> 'GaussianBeam':
        """Create a beam from spot size w and curvature R at the current plane."""
        inv_q_imag = -(wavelength / (np.pi * n * w**2))
        inv_q_real = 1.0 / R if not np.isinf(R) else 0.0
        
        denom = inv_q_real + 1j * inv_q_imag
        q = 1.0 / denom
        return cls(q=q, wavelength=wavelength)


    # ---- Debugging ----
    def __str__(self) -> str:
        """
        Human-readable representation assuming n=1.0 (Air/Vacuum).
        Shows current beam parameters and waist location.
        """
        # calculate parameters (assuming n=1 for display purposes)
        w_current = self.w_at_z(0, n=1.0)
        R_current = self.curvature_at_z(0)
        w_waist = self.w0(n=1.0)
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