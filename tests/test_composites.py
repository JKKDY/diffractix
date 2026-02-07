import pytest
import numpy as np
from diffractix.composites import ElementSequence, Slab, ThickLens
from diffractix.system import System
from diffractix.elements import Space, Interface
from diffractix.beams import GaussianBeam
from diffractix.graph import Parameter

# Helper for integration tests
def create_test_beam():
    return GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

# ==========================================
# 1. TEST BASE SEQUENCE LOGIC
# ==========================================

def test_sequence_behavior():
    """ElementSequence should behave like a list but with extra methods."""
    s1 = Space(d=10)
    s2 = Space(d=20)
    seq = ElementSequence([s1, s2])
    
    # List behavior
    assert len(seq) == 2
    assert seq[0] is s1
    assert seq[1] is s2
    
    # Iteration
    items = [x for x in seq]
    assert items == [s1, s2]


def test_sequence_variable_aliasing():
    """
    Verify that .variable('alias') triggers the correct internal parameters
    defined in the alias map.
    """
    # Setup: 2 Spaces
    s1 = Space(d=10, label="s1")
    s2 = Space(d=20, label="s2")
    
    # Define map: 'total_length' -> s1.d and s2.d
    aliases = {
        'total_length': [(0, 'd'), (1, 'd')]
    }
    seq = ElementSequence([s1, s2], aliases=aliases)
    
    # ACT: Mark 'total_length' as variable
    seq.variable('total_length')
    
    # ASSERT: Both sub-elements should now be variable
    # (Assuming Space uses Parameters internally and .variable() sets fixed=False)
    assert s1.d.fixed is False
    assert s2.d.fixed is False


def test_sequence_variable_fallback():
    """
    If a name is NOT in aliases, it should try to set it on all elements 
    that possess that parameter name.
    """
    s1 = Space(d=10, label="s1") # Has 'd'
    s2 = Space(d=20, label="s2") # Has 'd'
    
    # No aliases provided
    seq = ElementSequence([s1, s2])
    
    # ACT: 'd' is not an alias, so it falls back to checking param_names
    seq.variable('d')
    
    assert s1.d.fixed is False
    assert s2.d.fixed is False


# ==========================================
# 2. TEST SLAB COMPOSITE
# ==========================================

def test_slab_structure():
    """Verify a Slab expands into [Interface, Space, Interface]."""
    slab = Slab(d=10.0, n=1.5, n_ambient=1.0)
    
    assert len(slab) == 3
    
    # 1. Front Interface (Air -> Glass)
    front = slab[0]
    assert isinstance(front, Interface)
    assert front.n1.value == 1.0
    assert front.n2.value == 1.5
    
    # 2. Body (Glass)
    body = slab[1]
    assert isinstance(body, Space)
    assert body.n.value == 1.5
    assert body.d.value == 10.0
    
    # 3. Back Interface (Glass -> Air)
    back = slab[2]
    assert isinstance(back, Interface)
    assert back.n1.value == 1.5
    assert back.n2.value == 1.0


def test_slab_variables():
    """Verify 'd' and 'n' aliases work on Slab."""
    slab = Slab(d=10.0, n=1.5)
    
    # 1. Variable Thickness
    slab.variable('d')
    # Should target the middle Space element
    assert slab[1].d.fixed is False
    
    # 2. Variable Index
    slab.variable('n')
    # Should target n2 of front, n of body, n1 of back
    assert slab[0].n2.fixed is False # Front output
    assert slab[1].n.fixed is False  # Body
    assert slab[2].n1.fixed is False # Back input


# ==========================================
# 3. TEST THICK LENS COMPOSITE
# ==========================================

def test_thick_lens_geometry():
    """Verify ThickLens geometry setup."""
    tl = ThickLens(d=5.0, n=1.5, R1=100.0, R2=-100.0)
    
    # Front Interface R1
    assert tl[0].R.value == 100.0
    
    # Back Interface R2
    assert tl[2].R.value == -100.0
    
    # Check Aliases work
    tl.variable('R1')
    assert tl[0].R.fixed is False
    assert tl[2].R.fixed is True # R2 untouched


# ==========================================
# 4. SYSTEM INTEGRATION
# ==========================================

def test_composite_system_build():
    """
    Add a ThickLens to a System and verify it is correctly flattened 
    and compiled into the simulation.
    """
    sys = System(ambient_n=1.0)
    sys.add_input(create_test_beam())
    
    # Create Lens and mark Radii as optimizable
    tl = ThickLens(d=5.0, n=1.5, R1=50.0, R2=-50.0)
    tl.variable('R1', 'R2')
    
    # Add the composite (which is a list)
    # The System.add method handles Iterable duck-typing
    sys.add(tl)
    
    # Build
    sim = sys.build()
    
    # 1. Verify Flattening
    # [Interface, Space, Interface]
    assert len(sim.steps) == 3
    
    # 2. Verify Optimization Vector
    # Should contain R1 (50) and R2 (-50).
    # Note: 'd' and 'n' were not marked variable.
    assert len(sim.initial_values) == 2
    
    vals = sorted(sim.initial_values)
    assert np.allclose(vals, [-50.0, 50.0])


def test_composite_shared_parameter_instability_check():
    """
    Advanced: When we initialize Slab(n=1.5), the float 1.5 is passed to 3 different elements.
    By default, they create 3 DIFFERENT Parameter instances.
    Calls to .variable('n') unlock all 3.
    This test verifies that the system sees 3 independent variables (unless we share the instance).
    
    (Note: In a robust design, you might want them shared, but testing current behavior).
    """
    sys = System()
    sys.add_input(create_test_beam())
    
    slab = Slab(d=10.0, n=1.5)
    slab.variable('n') # Unlocks front.n2, body.n, back.n1
    
    sys.add(slab)
    sim = sys.build()
    
    # Since we passed a float '1.5', the elements created their own Parameters.
    # The alias unlocked all 3.
    # So the optimizer will see 3 variables with value 1.5.
    
    # Note: If the user explicitly passed a shared Parameter object:
    # p = Parameter(1.5, "n_shared")
    # slab = Slab(d=10, n=p)
    # Then we would see only 1 variable. 
    # But here we test the default float behavior.
    
    assert len(sim.initial_values) == 3
    assert np.allclose(sim.initial_values, [1.5, 1.5, 1.5])