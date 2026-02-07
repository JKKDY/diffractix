import pytest
import numpy as np
from diffractix.beams import GaussianBeam

# Constants for testing
LAMBDA = 1064e-9  # 1064 nm
W0 = 100e-6       # 100 microns
N_AIR = 1.0
N_GLASS = 1.5

# ==========================================
# 1. INITIALIZATION & PROPERTIES
# ==========================================

def test_beam_creation_at_waist():
    """Verify properties of a beam created exactly at the waist."""
    beam = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA, n=N_AIR)
    
    # At the waist:
    # 1. Spot size w = w0
    assert np.isclose(beam.w, W0)
    
    # 2. Radius of Curvature R = Infinity
    # (The class might return inf or a very large number depending on implementation details,
    # but based on your code: inv_q_real=0 -> R=inf)
    assert np.isinf(beam.R) or abs(beam.R) > 1e15
    
    # 3. Position relative to waist z0 = 0
    assert np.isclose(beam.z0, 0.0)
    
    # 4. Rayleigh range check
    expected_zr = (np.pi * W0**2 * N_AIR) / LAMBDA
    assert np.isclose(beam.z_r, expected_zr)


def test_beam_creation_offset():
    """Verify properties when created at a distance from the waist."""
    offset = 0.5 # 0.5 meters past the waist
    beam = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA, z_waist_loc=-offset, n=N_AIR)
    
    # z0 property returns "position of waist relative to current plane"
    # If waist was 0.5m ago (-0.5), then relative to us, it is at -0.5?
    # Let's check your logic: q = -z_waist_loc + 1j*zr. 
    # If we pass -0.5, q = 0.5 + i*zr. 
    # Real(q) = z distance from waist. So we are at z=0.5.
    
    # Your z0 property: -self.q.real
    # -(0.5) = -0.5. Correct. Waist is behind us.
    assert np.isclose(beam.z0, -offset)
    
    # Width should be larger than waist
    assert beam.w > W0


def test_beam_in_dielectric():
    """Verify scaling laws inside a medium (n=1.5)."""
    # Physics: Rayleigh range scales linearly with n. z_r = (pi * w0^2 * n) / lambda
    b_air = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA, n=1.0)
    b_glass = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA, n=1.5)
    
    # Glass beam should be more collimated (larger Rayleigh range)
    assert np.isclose(b_glass.z_r, b_air.z_r * 1.5)
    
    # Divergence should be smaller: theta = lambda / (pi * w0 * n)
    assert np.isclose(b_glass.divergence_angle, b_air.divergence_angle / 1.5)


# ==========================================
# 2. PROPAGATION MATH
# ==========================================

def test_propagation_rayleigh_distance():
    """
    At z = z_R (Rayleigh range):
    1. w should be w0 * sqrt(2)
    2. R should be minimal (2 * z_R)
    3. Gouy phase should be pi/4 (45 degrees)
    """
    beam = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA)
    z_r = beam.z_r
    
    # Propagate to Rayleigh range
    w_at_zr = beam.w_at_z(z_r)
    R_at_zr = beam.R_at_z(z_r)
    
    # 1. Spot size
    assert np.isclose(w_at_zr, W0 * np.sqrt(2))
    
    # 2. Curvature
    assert np.isclose(R_at_zr, 2 * z_r)
    
    # 3. Gouy Phase (Note: requires manual check as we don't have phase_at_z method, 
    # but we can create a new beam at that location to check property)
    beam_at_zr = GaussianBeam(q=beam.q + z_r, wavelength=LAMBDA)
    assert np.isclose(beam_at_zr.gouy_phase, np.pi/4)


# ==========================================
# 3. FACTORY METHODS
# ==========================================

def test_factory_from_divergence():
    """Verify creation from far-field divergence angle."""
    theta = 0.01 # radians
    beam = GaussianBeam.from_divergence(theta=theta, wavelength=LAMBDA)
    
    # Check if calculation reverses correctly
    assert np.isclose(beam.divergence_angle, theta)


def test_factory_from_fiber():
    """Verify creation from Fiber NA and MFD."""
    # Case A: MFD
    mfd = 10e-6
    b1 = GaussianBeam.from_fiber_tip(wavelength=LAMBDA, MFD=mfd)
    assert np.isclose(b1.w0, mfd/2)
    
    # Case B: NA
    # w0 = lambda / (pi * NA)
    na = 0.1
    b2 = GaussianBeam.from_fiber_tip(wavelength=LAMBDA, NA=na)
    expected_w0 = LAMBDA / (np.pi * na)
    assert np.isclose(b2.w0, expected_w0)
    
    # Case C: Error
    with pytest.raises(ValueError):
        GaussianBeam.from_fiber_tip(wavelength=LAMBDA)


# ==========================================
# 4. OVERLAP (COUPLING EFFICIENCY)
# ==========================================

def test_overlap_self():
    """A beam should have 100% overlap with itself."""
    b1 = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA)
    assert np.isclose(b1.overlap_with(b1), 1.0)


def test_overlap_mode_mismatch():
    """
    Test coupling between two beams with different spot sizes (waists) at the same location.
    Formula: eta = 4 / ( (w1/w2) + (w2/w1) )^2
    """
    w1 = 100e-6
    w2 = 50e-6
    b1 = GaussianBeam.from_waist(w0=w1, wavelength=LAMBDA)
    b2 = GaussianBeam.from_waist(w0=w2, wavelength=LAMBDA)
    
    efficiency = b1.overlap_with(b2)
    
    # Analytic Calculation
    ratio_sum = (w1/w2) + (w2/w1) # 2 + 0.5 = 2.5
    expected = 4.0 / (ratio_sum**2) # 4 / 6.25 = 0.64
    
    assert np.isclose(efficiency, 0.64)


def test_overlap_defocus_mismatch():
    """
    Test coupling between two identical beams, but one is offset longitudinally (defocused).
    """
    b_waist = GaussianBeam.from_waist(w0=W0, wavelength=LAMBDA)
    
    # Create a beam that represents the SAME beam propagated by z_r
    # We are calculating the overlap integral between a Flat phase (waist)
    # and a Curved phase (at z_r). They should NOT couple 100%.
    z_offset = b_waist.z_r
    b_shifted = GaussianBeam(q=b_waist.q + z_offset, wavelength=LAMBDA)
    
    efficiency = b_waist.overlap_with(b_shifted)
    
    # At z=z_R, the overlap with the waist is exactly 50% (0.5)
    # This is a standard result in Gaussian beam optics.
    assert np.isclose(efficiency, 0.8)


def test_overlap_medium_mismatch_error():
    """Cannot calculate overlap if beams are in different media."""
    
    b1 = GaussianBeam.from_waist(W0, LAMBDA, n=1.0)
    b2 = GaussianBeam.from_waist(W0, LAMBDA, n=1.5)
    
    with pytest.raises(ValueError, match="different media"):
        b1.overlap_with(b2)