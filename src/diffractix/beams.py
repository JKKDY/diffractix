
from dataclasses import dataclass
from typing import Tuple
import autograd.numpy as np
import cmath


@dataclass
class GaussianBeam:
    """
    Represents a Gaussian Beam at a specific point in space and medium.
    
    Attributes:
        q (complex): The complex beam parameter describing curvature and width.
        wavelength (float): The vacuum wavelength of the light.
        n (float): The refractive index of the medium the beam is currently in.
    """

    q: complex
    wavelength: float
    n: float = 1.0


    #-----------
    # PROPERTIES
    #-----------
    @property
    def w(self) -> float:
        """Spot size radius (1/e^2 intensity) at the current position."""
        val_w, _ = self.w_R_from_q(self.q, self.wavelength, self.n)
        return val_w

    @property
    def R(self) -> float:
        """Radius of curvature at the current position."""
        _, val_R = self.w_R_from_q(self.q, self.wavelength, self.n)
        return val_R


    @property
    def z_r(self) -> float:
        """Rayleigh range (in terms of propagation distance z)."""
        # q = z + i*z_r
        return np.imag(self.q)

    @property
    def z0(self) -> float:
        """
        Position of the waist relative to current plane.
        (+) Waist is in front (Converging). (-) Waist is behind (Diverging).
        """
        # q = (z - z_waist) + i*z_r
        # Real(q) = z - z_waist. If we define current z=0, then Real(q) = -z_waist.
        return -self.q.real

    @property
    def w0(self) -> float:
        """Calculate waist radius w0 (minimum radius the beam reaches at its focal point)"""
        # z_r = (pi * w0^2 * n) / lambda
        return np.sqrt(self.z_r * self.wavelength / (np.pi * self.n))

    @property
    def divergence_angle(self) -> float:
        """Far-field divergence half-angle (radians)."""
        return self.wavelength / (np.pi * self.w0 * self.n)

    @property
    def NA(self) -> float:
        """
        Numerical Aperture (NA). 
        Approximation: NA ≈ n * theta (for small angles).
        Useful for fiber coupling or objective lens matching.
        """
        # theta = lambda / (pi * w0 * n)
        # NA = n * theta = lambda / (pi * w0)
        return self.wavelength / (np.pi * self.w0)

    @property
    def gouy_phase(self) -> float:
        """
        The Gouy phase shift psi(z) at the current position relative to the waist.
        psi = arctan(z / z_r)
        Range: -pi/2 to +pi/2
        """
        # Re(q) is the distance z from the waist. Im(q) is z_r.
        return np.arctan2(np.real(self.q), np.imag(self.q))

    @property
    def b(self) -> float:
        """The Confocal Parameter (twice the Rayleigh range)."""
        return 2 * self.z_r
    

    #--------------------
    # CALCULATION METHODS
    #--------------------
    def R_at_z(self, z: float) -> float:
        """Radius of curvature R at distance z."""
        q_z = self.q + z
        _, R_new = self.w_R_from_q(q_z, self.wavelength, self.n) 
        return R_new

    def w_at_z(self, z: float, n: float = None) -> float:
        """Calculate spot size w at distance z from current plane."""
        q_z = self.q + z
        w_new, _ = self.w_R_from_q(q_z, self.wavelength, self.n)
        return w_new

    def overlap_with(self, other: 'GaussianBeam') -> float:
        """
        Calculates the power coupling efficiency (0.0 to 1.0) between this beam 
        and another Gaussian beam (e.g., a fiber mode).
        
        Formula: 4 * |(1/q1_conj + 1/q2)|^-2 * Im(1/q1) * Im(1/q2) 
        (Requires both beams to be at the same z position and in the same n)
        """
        if abs(self.n - other.n) > 1e-6:
             raise ValueError("Cannot calculate overlap between beams in different media.")

        # Using the q-parameter mismatch formula for coupling efficiency
        q1 = self.q
        q2 = other.q
        
        # Mismatch factor
        # This is a generalized formula for fundamental mode overlap
        # eta = 4 / ( abs( (w1/w2) + (w2/w1) )^2 + ( pi*w1*w2/lambda * (1/R1 - 1/R2) )^2 )
        # But computing via complex q is faster:
        
        # Power Coupling = 4 / ( |(w1/w2) + (w2/w1)|^2 + ... )
        # A simpler complex form:
        # eta = 4 * Im(1/q1) * Im(1/q2) / | 1/q1* + 1/q2 |^2
        # Note: q1* is complex conjugate.
        
        inv_q1 = 1.0 / q1
        inv_q2 = 1.0 / q2

        inv_q1_imag = np.imag(inv_q)
        inv_q2_imag = np.imag(inv_q)
        
        numerator = 4 * inv_q1_imag * inv_q2_imag
        denominator = abs(np.conj(inv_q1) + inv_q2)**2
        
        return numerator / denominator

    @staticmethod
    def w_R_from_q(q: complex, wavelength: float, n: float = 1.0) -> tuple[float, float]:
        """
        Extracts beam width (w) and radius of curvature (R) from the complex q-parameter.
        
        Math: 1/q = 1/R - i * (lambda / (pi * n * w^2))
        """
        inv_q = 1.0 / q

        # need to use np.real/imag to ensure autograd compatibility
        inv_q_real = np.real(inv_q)
        inv_q_imag = np.imag(inv_q)
        
        # radius of curvature (real part of 1/q)
        R = np.inf if abs(inv_q_real) < 1e-16 else 1.0 / inv_q_real
            
        # Beam waist 
        # Im(1/q) = - lambda / (pi * n * w^2)  =>  w = sqrt( -lambda / (pi * n * Im(1/q)) )        
        # safety check: q.imag should always be negative for a valid Gaussian beam (forward propagation)
        if inv_q_imag > -1e-16: 
            raise Exception("Invalid q-parameter: Imaginary part of 1/q must be negative for a physical Gaussian beam.")

        w_sq = -wavelength / (np.pi * n * inv_q_imag)
        w = np.sqrt(w_sq)
        
        return w, R


    #----------------
    # FACTORY METHODS
    #----------------
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
        return cls(q=q, wavelength=wavelength, n=n)

    @classmethod
    def from_w_and_R(cls, w: float, R: float, wavelength: float, n: float = 1.0) -> 'GaussianBeam':
        """Create a beam from spot size w and curvature R at the current plane."""
        inv_q_imag = -(wavelength / (np.pi * n * w**2))
        inv_q_real = 1.0 / R if not np.isinf(R) else 0.0
        
        denom = inv_q_real + 1j * inv_q_imag
        q = 1.0 / denom
        return cls(q=q, wavelength=wavelength, n=n)

    @classmethod
    def from_divergence(cls, theta: float, wavelength: float, n: float = 1.0) -> 'GaussianBeam':
        """
        Create a beam defined by its far-field divergence half-angle (in radians).
        Assumes the beam starts AT the waist (z=0, R=inf).
        """
        # theta = lambda / (pi * w0 * n)  ->  w0 = lambda / (pi * theta * n)
        w0 = wavelength / (np.pi * theta * n)
        return cls.from_waist(w0, wavelength, z_waist_loc=0.0, n=n)


    #----------
    # DEBUGGING
    #----------
    def __str__(self) -> str:
        """
        Human-readable representation assuming n=1.0 (Air/Vacuum).
        Shows current beam parameters and waist location.
        """
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
        """
        Debug plot: Visualizes the beam's envelope relative to the current position.
        
        Args:
            span (float): Total propagation distance to plot centered on the current position.
                          If None, defaults to 4 * Rayleigh Range or 4 * z_waist (whichever is larger).
            points (int): Number of plotting points.
        """
        import matplotlib.pyplot as plt

        # We want to show enough context. If the waist is 1km away, showing 1mm is useless.
        # If the waist is 1mm away, showing 1km is useless.
        if span is None:
            # Heuristic: Look at distance to waist (z0) and Rayleigh range (z_r)
            dist_to_waist = abs(self.z0)
            scale = max(dist_to_waist, self.z_r)
            span = 4 * scale if scale > 0 else 0.1 # Default to 10cm if scale is tiny

        z_local = np.linspace(-span/2, span/2, points)
        
        # Calculate Envelope w(z):
        w_vals = [self.w_at_z(z) for z in z_local]
        w_vals = np.array(w_vals)

        # Ploting
        plt.figure(figsize=(8, 4))
        
        # The Envelope
        plt.fill_between(z_local, w_vals, -w_vals, color='blue', alpha=0.1, label='Beam Envelope (1/e²)')
        plt.plot(z_local, w_vals, 'b-', lw=1.5)
        plt.plot(z_local, -w_vals, 'b-', lw=1.5)
        
        # Current Position Marker
        plt.axvline(0, color='red', linestyle='--', label='Current Position')
        plt.text(0, max(w_vals)*0.1, " You Are Here", color='red', rotation=90, verticalalignment='bottom')

        # Waist Position Marker (if within range)
        z_waist_rel = self.z0 # z0 is "position of waist relative to us"
        if min(z_local) < z_waist_rel < max(z_local):
            plt.axvline(z_waist_rel, color='green', linestyle=':', label='Waist Location')
            plt.text(z_waist_rel, max(w_vals)*0.1, f" Waist (w0={self.w0*1e6:.1f}um)", color='green', rotation=90)

        # Labels
        plt.title(f"Beam Diagnostic (n={self.n})")
        plt.xlabel(f"Relative Propagation Distance [m]\n(z > 0 is forward)")
        plt.ylabel("Beam Radius w(z) [m]")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()
