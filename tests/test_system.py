import pytest
import numpy as np
from diffractix.system import System
from diffractix.elements import Space, ThinLens, Interface, ABCD
from diffractix.graph import Parameter, Constant


#-----------------------
# TEST LAYOUT VALIDATION
# ----------------------
def test_validate_physics_continuity_success():
    """Test a valid sequence: Space -> Interface -> Space."""
    sys = System(ambient_n=1.0)
    
    # 1. Travel in Air (n=1)
    sys.add(Space(d=10, n=1.0))
    
    # 2. Hit Surface (1 -> 1.5)
    # This UPDATES current_n to 1.5
    sys.add(Interface(n1=1.0, n2=1.5, R=10))
    
    # 3. Travel in Glass (n=1.5)
    # Matches current_n (1.5). Valid.
    sys.add(Space(d=5, n=1.5))
    
    sys._bind_environment_variables(sys.elements)
    sys._validate_layout() # Should pass

def test_validate_physics_transparent_skip():
    """
    Test that ThinLens is treated as transparent (ignored).
    Space(n=1) -> Lens -> Space(n=1) is valid.
    """
    sys = System(ambient_n=1.0)
    
    sys.add(Space(d=10, n=1.0))
    sys.add(ThinLens(f=50))  # L=0, Ignored. current_n remains 1.0
    sys.add(Space(d=10, n=1.0)) # Matches 1.0. Valid.
    
    sys._bind_environment_variables(sys.elements)
    sys._validate_layout()

def test_validate_physics_passthrough_error():
    """
    Test that a mismatch is caught even if a Lens is in the way.
    Space(n=1) -> Lens -> Space(n=1.5) is INVALID.
    """
    sys = System(ambient_n=1.0)
    
    sys.add(Space(d=10, n=1.0)) # current_n = 1.0
    sys.add(ThinLens(f=50))     # Ignored. current_n = 1.0
    
    # Space expects 1.5, but current_n is 1.0.
    sys.add(Space(d=10, n=1.5)) 
    
    sys._bind_environment_variables(sys.elements)

    with pytest.raises(ValueError, match="Physics Violation"):
        sys._validate_layout()

def test_validate_abcd_override():
    """Test that ABCD is allowed to forcefully reset the index."""
    sys = System(ambient_n=1.0)
    sys.add(Space(d=10, n=1.0))
    
    # Black Box: Takes n=1.0, outputs n=1.5
    sys.add(ABCD(matrix_val=np.eye(2), thickness=0, n=1.5))
    
    # Next space matches the new index
    sys.add(Space(d=10, n=1.5)) 
    
    sys._bind_environment_variables(sys.elements)
    sys._validate_layout()


#--------------------
# TEST RESOLVE LAYOUT
# -------------------


def test_resolve_layout_hard_constraint_wiring():
    """
    Verify that resolving layout creates a hard graph link 
    between upstream variables and the autospacer.
    """
    sys = System()
    
    # 1. Variable Element (Degree of Freedom)
    d_var = Parameter(10.0, "d_var", fixed=False)
    sys.add(Space(d=d_var))
    
    # 2. Fixed Anchor at z=100
    sys.add(ThinLens(f=50), z=100.0)
    
    # --- UNIT TEST SETUP ---
    # We only run the specific steps required to generate _compiled_elements
    # sys._bind_environment_variables(sys.elements) # (Assuming this exists/is needed)
    sys._resolve_layout() 
    
    # 3. Verify Internal State
    # List should be: [Space(d_var), AutoSpace(gap), ThinLens]
    assert len(sys._compiled_elements) == 3
    
    spacer = sys._compiled_elements[1]
    assert isinstance(spacer, Space)
    assert "AutoSpace" in spacer.label
    
    # 4. Verify Value (Initial State)
    # 100 - 10 = 90
    assert spacer.length == 90.0
    
    # 5. Verify Wiring (The Magic)
    # Changing the upstream variable should INSTANTLY update the spacer
    # purely via graph connection, without re-running resolve_layout
    d_var.value = 40.0
    
    # The spacer should now be 100 - 40 = 60
    assert spacer.length == 60.0


def test_resolve_layout_optimizable_anchor():
    """
    Verify that if optimize_z=True, the 'Target Z' itself becomes a variable parameter.
    Spacer = (Variable Target) - (Upstream Fixed)
    """
    sys = System()
    
    # Upstream is fixed (d=10)
    sys.add(Space(d=10.0))
    
    # Anchor is at 100, but we want to optimize that 100.
    sys.add(ThinLens(f=50), z=100.0, optimize_z=True)
    
    # --- UNIT TEST SETUP ---
    sys._resolve_layout()
    
    spacer = sys._compiled_elements[1]
    
    # 1. Spacer length check: 100 - 10 = 90
    assert spacer.length == 90.0
    
    # 2. Check the graph dependency
    # The spacer's length node should depend on a variable (the anchor)
    # verifying that it is NOT constant.
    assert spacer.element_length.is_constant is False
    

def test_resolve_layout_multiple_anchors():
    """
    Verify chained absolute positions work correctly.
    0 -> [Space 10] -> [Auto 90] -> Lens@100 -> [Auto 50] -> Mirror@150
    """
    sys = System()
    sys.add(Space(d=10))             # Ends at 10
    sys.add(ThinLens(f=50), z=100)   # Ends at 100 (Spacer=90)
    sys.add(ThinLens(f=50), z=150)   # Ends at 150 (Spacer=50)
    
    # --- UNIT TEST SETUP ---
    sys._resolve_layout()
    
    # Check elements list
    # [Space(10), Auto(90), Lens, Auto(50), Lens]
    assert len(sys._compiled_elements) == 5
    
    assert sys._compiled_elements[1].length == 90.0
    assert sys._compiled_elements[3].length == 50.0


def test_resolve_layout_start_anchor():
    """
    Verify placing an element at z > 0 as the very first item works.
    Should insert a spacer at the beginning.
    """
    sys = System()
    sys.add(ThinLens(f=50), z=20.0)
    
    # --- UNIT TEST SETUP ---
    sys._resolve_layout()
    
    # Should be [AutoSpace(20), ThinLens]
    assert len(sys._compiled_elements) == 2
    assert isinstance(sys._compiled_elements[0], Space)
    assert sys._compiled_elements[0].length == 20.0