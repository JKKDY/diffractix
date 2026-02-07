import pytest
import numpy as np
from diffractix.system import System
from diffractix.elements import Space, ThinLens, Interface, GaussianAperture
from diffractix.beams import GaussianBeam
from autograd import grad

# helper
def create_basic_sim():
    """
    Creates a simple pre-built simulation: 
    Input(w0=1mm) -> Space(1m)
    """
    sys = System(ambient_n=1.0)
    # 1064nm light, 1mm waist
    beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)
    sys.add_input(beam)
    
    sys.add(Space(d=1.0))
    
    return sys.build()


#-----------------------
# PHYSICS ACCURACY TESTS
#-----------------------

def test_simulation_run_free_space_expansion():
    """
    Verify that a beam expands correctly in free space (Vacuum).
    Math: w(z) = w0 * sqrt(1 + (z/zR)^2)
    """
    sim = create_basic_sim()
    result = sim.run()
    
    start_beam = result.trace[0][1]
    end_beam = result.trace[1][1]
    
    # Start: w = 1mm
    assert np.isclose(start_beam.w, 1e-3)
    
    # End: Calculate expected expansion
    z = 1.0
    zr = start_beam.z_r
    expected_w = 1e-3 * np.sqrt(1 + (z/zr)**2)
    
    assert np.isclose(end_beam.w, expected_w)

def test_simulation_medium_expansion():
    """
    Verify expansion inside a dielectric (n=1.5).
    Physics: Inside a medium, the effective Rayleigh range INCREASES (zR_n = zR_vac * n).
    The beam should diverge SLOWER than in air.
    """
    sys = System(ambient_n=1.5) # Entire system in glass
    beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9, n=1.5)
    sys.add_input(beam)
    
    sys.add(Space(d=1.0, n=1.5))
    sim = sys.build()
    result = sim.run()
    
    end_beam = result.trace[1][1]
    
    # Calculate expected w in medium
    # z_r_medium = (pi * w0^2 * n) / lambda
    z_r_medium = (np.pi * (1e-3)**2 * 1.5) / 1064e-9
    expected_w = 1e-3 * np.sqrt(1 + (1.0/z_r_medium)**2)
    
    assert np.isclose(end_beam.w, expected_w)
    
    # Sanity check: Should be smaller than air expansion
    # (Air expansion calculated manually for comparison)
    z_r_air = z_r_medium / 1.5
    w_air = 1e-3 * np.sqrt(1 + (1.0/z_r_air)**2)
    assert end_beam.w < w_air

def test_simulation_lens_focusing():
    """
    Verify that a lens focuses the beam.
    Setup: Waist -> Lens(f=0.5) -> Space(0.5)
    """
    sys = System()
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9))
    
    sys.add(ThinLens(f=0.1)) 
    sys.add(Space(d=0.1))    
    
    sim = sys.build()
    res = sim.run()
    
    beam_after_lens = res.trace[1][1]
    beam_end = res.trace[2][1]
    
    # Check curvature induction: 1/R_out = -1/f
    assert np.isclose(beam_after_lens.R, -0.1)
    
    # Check convergence: Spot size should decrease
    assert beam_end.w < beam_after_lens.w

def test_simulation_refractive_index_tracking():
    """
    Verify that the simulation correctly tracks the refractive index 'n'
    across interfaces and updates the beam object accordingly.
    """
    sys = System(ambient_n=1.0)
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6))
    
    # Air -> Interface -> Glass(1.5) -> Space -> Interface -> Air
    sys.add(Interface(n1=1.0, n2=1.5))
    sys.add(Space(d=0.1, n=1.5))
    
    sim = sys.build()
    res = sim.run()
    
    # Trace items: 
    # 0: Input (n=1.0)
    # 1: After Interface (n=1.5)
    # 2: End of Space (n=1.5)
    
    beam_input = res.trace[0][1]
    beam_in_glass = res.trace[1][1]
    beam_end_glass = res.trace[2][1]
    
    assert beam_input.n == 1.0
    assert beam_in_glass.n == 1.5
    assert beam_end_glass.n == 1.5
    
    # Verify q-parameter continuity check at interface
    # q_glass = q_air * (n_glass / n_air)
    # At the interface (z=0), q is purely imaginary i*zR
    # So zR_glass should be zR_air * 1.5
    assert np.isclose(beam_in_glass.z_r, beam_input.z_r * 1.5)





#-----------------------
# OPTIMIZER & DATA TESTS
#-----------------------

def test_simulation_optimizer_output_shape():
    """Verify run_for_optimizer returns a raw numpy array (Steps, 3)."""
    sim = create_basic_sim()
    data = sim.run_for_optimizer() # Use initial values
    
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3) # [z, w, R]
    
    # Check z=0, w=1mm
    assert data[0, 0] == 0.0
    assert np.isclose(data[0, 1], 1e-3)

def test_simulation_parameter_injection():
    """Verify that passing new parameters changes the simulation output."""
    sys = System()
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6))
    
    s1 = Space(d=1.0)
    s1.variable('d')
    sys.add(s1)
    
    sim = sys.build()
    
    # Run 1: d=1.0
    res_1 = sim.run_for_optimizer(sim.initial_values)
    assert res_1[-1, 0] == 1.0
    
    # Run 2: d=5.0
    new_params = np.array([5.0])
    res_2 = sim.run_for_optimizer(new_params)
    assert res_2[-1, 0] == 5.0
    
    # Physics check: Diffraction should be larger at z=5
    assert res_2[-1, 1] > res_1[-1, 1]

def test_simulation_aperture_transmission():
    """
    Test Gaussian Aperture transmission.
    Note: Aperture affects intensity/w, but in ABCD matrix theory 
    it makes q complex. This tests if the matrix multiplication handles complex C.
    """
    sys = System()
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6))
    
    # Add aperture with radius same as beam waist
    sys.add(GaussianAperture(a=1e-3))
    
    sim = sys.build()
    res = sim.run()
    
    beam_out = res.trace[1][1]
    
    # Physics: Transmission through gaussian aperture
    # 1/w_out^2 = 1/w_in^2 + 1/a^2
    # w_out = w_in / sqrt(2) if a == w_in
    expected_w = 1e-3 / np.sqrt(2)
    
    assert np.isclose(beam_out.w, expected_w)



#---------------
# AUTOGRAD TESTS
#---------------

def test_autograd_end_to_end_gradient():
    """
    Verify that we can compute gradients of the output 
    with respect to the input parameters. 
    This confirms no logical breaks in the computation graph.
    """
    # 1. Setup a simple focusing system
    sys = System(ambient_n=1.0)
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6))
    
    # Variable Lens
    l1 = ThinLens(f=0.1)
    l1.variable('f')
    sys.add(l1)
    
    # Fixed Space
    sys.add(Space(d=0.1))
    
    sim = sys.build()
    
    # 2. Define a Loss Function
    # Target: Minimize spot size w at the end
    def loss_function(theta):
        # theta is the vector of variables (f)
        # run_for_optimizer returns [z, w, R] array
        results = sim.run_for_optimizer(theta)
        final_w = results[-1, 1] 
        return final_w

    # 3. Compute Gradient
    # If the graph is broken, this will raise an error
    grad_fn = grad(loss_function)
    
    initial_theta = sim.initial_values
    gradient = grad_fn(initial_theta)
    
    # 4. Validation
    assert gradient is not None
    assert len(gradient) == 1
    # Physics check: Increasing f (weaker lens) should increase spot size 
    # (since we are near focus). So gradient should be positive?
    # Actually depends on z vs f relation, but simply checking it's non-zero 
    # and a float is usually enough for structural verification.
    assert isinstance(gradient[0], float)
    assert gradient[0] != 0.0

def test_autograd_with_composite_variables():
    """Verify gradients propagate through composite logic (Slab)."""
    sys = System()
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6))
    
    # Variable Slab Thickness
    from diffractix.composites import Slab
    slab = Slab(d=0.1, n=1.5)
    slab.variable('d')
    sys.add(slab)
    
    sim = sys.build()
    
    def loss(theta):
        res = sim.run_for_optimizer(theta)
        # Return final Gouy phase or similar (using complex q if possible)
        # For now, just spot size
        return res[-1, 1] 

    grad_fn = grad(loss)
    g = grad_fn(sim.initial_values)
    
    assert g is not None
    assert g[0] != 0.0