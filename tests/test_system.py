import pytest
import numpy as np
from diffractix.system import System
from diffractix.elements import Space, ThinLens, ABCD, Interface
from diffractix.graph import Parameter, Constant, Symbol, InputNode
from diffractix.beams import GaussianBeam

# ==========================================
# 1. TEST BINDING & ENVIRONMENT
# ==========================================

def test_bind_environment_variables():
    """
    Verify that symbols named "ambient_n" in elements are correctly 
    bound to the System's environment parameter.
    """
    # 1. Setup System with specific ambient index
    sys = System(ambient_n=1.33)

    space = Space(d=3, n=Symbol("ambient_n"))

    sys.add(space)

    # 3. Action
    sys._bind_environment_variables(sys.elements)

    # 4. Verify Binding
    # The symbol inside the InputNode should now point to sys.environment.ambient_n
    symbol = space.n.node
    assert isinstance(symbol, Symbol)
    assert symbol.name == "ambient_n"
    assert symbol.value == 1.33
    
    # Ensure it's a reference, not just a value copy
    sys.environment.ambient_n.value = 1.5
    assert symbol.value == 1.5


# ==========================================
# 2. TEST VALIDATION (GEOMETRY CHECKS)
# ==========================================

def test_validate_layout_success_relative():
    """Simple relative chain should always pass validation."""
    sys = System()
    sys.add(Space(d=10))
    sys.add(ThinLens(f=50))
    sys.add(Space(d=10))
    
    sys._bind_environment_variables(sys.elements)
    sys._validate_layout(sys.elements) # Should not raise


def test_validate_layout_success_absolute():
    """Valid absolute positioning should pass."""
    sys = System()
    sys.add(Space(d=10))           # z=10
    sys.add(ThinLens(f=50), z=20)  # z=20 (Gap=10)
    
    sys._bind_environment_variables(sys.elements)
    sys._validate_layout(sys.elements)


def test_validate_layout_overlap_error():
    """
    Error when absolute position requires negative space.
    Space(10) -> Lens(z=5).
    """
    sys = System()
    sys.add(Space(d=10.0)) # Ends at z=10
    
    # User tries to place Lens at z=5
    sys.add(ThinLens(f=50), z=5.0)
    
    sys._bind_environment_variables(sys.elements)
    
    with pytest.raises(ValueError, match="Requested absolute z=5.0000"):
        sys._validate_layout(sys.elements)


def test_validate_layout_monotonic_error():
    """
    Error when absolute positions are not strictly increasing (or equal).
    Lens(z=10) -> Lens(z=5)
    """
    sys = System()
    sys.add(ThinLens(f=50), z=10.0)
    sys.add(ThinLens(f=50), z=5.0) # Backwards!
    
    sys._bind_environment_variables(sys.elements)
    
    with pytest.raises(ValueError, match="Requested absolute z=5.0000"):
        sys._validate_layout(sys.elements)


# ==========================================
# 3. TEST RESOLUTION (GRAPH WIRING)
# ==========================================

def test_resolve_layout_auto_spacer_generation():
    """
    Verify that _resolve_layout inserts a Space element 
    to bridge the gap to an absolute position.
    """
    sys = System()
    sys.add(Space(d=10.0))          # 0 -> 10
    sys.add(ThinLens(f=50), z=30.0) # 10 -> 30 (Gap=20)
    
    sys._bind_environment_variables(sys.elements)
    resolved = sys._resolve_layout(sys.elements)
    
    # Expect: [Space(10), AutoSpace(20), ThinLens]
    assert len(resolved) == 3
    
    spacer = resolved[1]
    assert isinstance(spacer, Space)
    assert "AutoSpace" in spacer.label
    assert spacer.length == 20.0


def test_resolve_layout_hard_constraint_wiring():
    """
    CRITICAL TEST: Verify that the AutoSpace is hard-wired to upstream variables.
    If we change the upstream space length, the AutoSpace should shrink/grow 
    automatically to maintain the absolute target.
    """
    sys = System()
    
    # 1. Variable Upstream Element
    d_var = Parameter(10.0, "d_var", fixed=False)
    sys.add(Space(d=d_var))
    
    # 2. Fixed Anchor at z=100
    # AutoSpace = 100 - d_var
    sys.add(ThinLens(f=50), z=100.0)
    
    # Resolve
    sys._bind_environment_variables(sys.elements)
    resolved = sys._resolve_layout(sys.elements)
    spacer = resolved[1]
    
    # 3. Initial Check (100 - 10 = 90)
    assert spacer.length == 90.0
    
    # 4. Modify Upstream Variable (simulate optimizer step)
    d_var.value = 40.0
    
    # 5. Check wiring
    # The spacer should INSTANTLY update because its 'd' is an AST BinaryOp
    # 100 - 40 = 60
    assert spacer.length == 60.0


def test_resolve_layout_optimizable_anchor():
    """
    Verify that if optimize_z=True, the Target Position itself becomes a variable.
    Spacer = (Variable_Target) - (Upstream_Fixed)
    """
    sys = System()
    sys.add(Space(d=10.0))
    
    # We want to optimize the position of this lens, starting at z=100
    sys.add(ThinLens(f=50), z=100.0, optimize_z=True)
    
    sys._bind_environment_variables(sys.elements)
    resolved = sys._resolve_layout(sys.elements)
    spacer = resolved[1]
    
    # 1. Check Initial Length (100 - 10 = 90)
    assert spacer.length == 90.0
    
    # 2. Check Graph Structure
    # The spacer's d node should depend on a Parameter named "Anchor_..."
    # We can't easily grab the anchor parameter reference unless we traverse the graph,
    # but we can verify that the spacer is NOT constant.
    assert spacer.element_length.is_constant is False
    
    # 3. Verify 'optimize_z' effect logic
    # To be sure, we can inspect the inputs of the spacer's d node
    # It should be [AnchorNode, UpstreamNode]
    # (This assumes BinaryOp structure, implementation detail check)
    d_node = spacer.d.node
    assert isinstance(d_node.left, Parameter) # The Anchor
    assert d_node.left.name.startswith("Anchor_")
    assert d_node.left.fixed is False # It is optimizable


def test_resolve_layout_start_anchor():
    """
    Verify absolute positioning works for the very first element.
    z=20 -> Insert Spacer(20) at start.
    """
    sys = System()
    sys.add(ThinLens(f=50), z=20.0)
    
    sys._bind_environment_variables(sys.elements)
    resolved = sys._resolve_layout(sys.elements)
    
    # Expect: [AutoSpace(20), ThinLens]
    assert len(resolved) == 2
    assert resolved[0].length == 20.0


# ==========================================
# TEST REFRACTIVE INDEX RESOLUTION
# ==========================================

def test_resolve_indices_inheritance_chain():
    """
    Scenario: Ambient(1.33) -> Space(inherit) -> Space(inherit)
    Both spaces should bind to the ambient index node.
    """
    sys = System(ambient_n=1.33)
    
    # Standard Spaces default to n=InputNode(None) [Inherit]
    s1 = Space(d=10.0)
    s2 = Space(d=20.0)
    
    elements = [s1, s2]
    
    # --- ACT ---
    # We must bind environment variables first (if any exist)
    sys._bind_environment_variables(elements)
    resolved = sys._resolve_refractive_indices(elements)
    
    # --- ASSERT ---
    # 1. Structure: List length unchanged
    assert len(resolved) == 2
    
    # 2. Binding: s1.n should be bound to ambient_n
    # Note: Accessing .value works only if bound
    assert s1.element_refractive_index.value == 1.33
    
    # 3. Wiring: Verify they point to the exact same AST node (the ambient parameter)
    # This confirms the graph is wired, not just copied values
    assert s1.element_refractive_index.node is sys.environment.ambient_n
    assert s2.element_refractive_index.node is sys.environment.ambient_n


def test_resolve_indices_fixed_mismatch_error():
    """
    Scenario: Ambient(1.0) -> Space(n=1.5)
    Should raise ValueError because of physical discontinuity.
    """
    sys = System(ambient_n=1.0)
    
    # Space with Explicit Index
    s1 = Space(d=10.0, n=1.5)
    elements = [s1]
    
    sys._bind_environment_variables(elements)

    with pytest.raises(ValueError, match="Refractive Index Mismatch"):
        sys._resolve_refractive_indices(elements, auto_insert_interface=False)


def test_resolve_indices_auto_insert_interface():
    """
    Scenario: Ambient(1.0) -> Space(n=1.5) with auto_insert_interface=True
    Should insert an Interface(1.0->1.5) automatically.
    """
    sys = System(ambient_n=1.0)
    s1 = Space(d=10.0, n=1.5)
    elements = [s1]
    
    sys._bind_environment_variables(elements)
    resolved = sys._resolve_refractive_indices(elements, auto_insert_interface=True)
    
    # --- ASSERT ---
    # Expect: [Interface, Space]
    assert len(resolved) == 2
    
    # Check inserted element
    iface = resolved[0]
    assert isinstance(iface, Interface)
    assert np.isclose(iface.n1.value, 1.0)
    assert np.isclose(iface.n2.value, 1.5)
    
    # Check original element remains
    assert resolved[1] is s1


def test_resolve_indices_passthrough_lens():
    """
    Scenario: Ambient(1.0) -> ThinLens -> Space(n=1.5)
    The Lens should pass the 1.0 flow through. The subsequent Space(1.5) 
    should trigger a mismatch error (because it sees 1.0 coming from the Lens).
    """
    sys = System(ambient_n=1.0)
    
    lens = ThinLens(f=50.0) # Transparent / Inherit
    s1 = Space(d=10.0, n=1.5) # Explicit 1.5
    
    elements = [lens, s1]
    sys._bind_environment_variables(elements)
    
    # Should fail at the Space, not the Lens
    with pytest.raises(ValueError, match="Refractive Index Mismatch"):
        sys._resolve_refractive_indices(elements)


def test_resolve_indices_blackbox_override():
    """
    Scenario: Ambient(1.0) -> ABCD(n=1.5) -> Space(inherit)
    The ABCD matrix overrides the flow to 1.5. The following Space inherits 1.5.
    """
    sys = System(ambient_n=1.0)
    
    # Blackbox defined to output n=1.5
    # (Assuming ABCD constructor takes 'n' as the output index)
    bb = ABCD(matrix_val=[[1,0],[0,1]], n=1.5) 
    
    s1 = Space(d=10.0) # Inherit
    
    elements = [bb, s1]
    sys._bind_environment_variables(elements)
    resolved = sys._resolve_refractive_indices(elements)
    
    # Space should inherit 1.5 from the BlackBox
    assert s1.element_refractive_index.value == 1.5
    
    # Verify Wiring: s1 is bound to bb's output node
    # (Assuming bb.n wraps the value in an InputNode/Constant)
    # The resolution logic sets current_index_node = bb.element_refractive_index
    # s1 binds to current_index_node.
    assert s1.element_refractive_index.node is bb.element_refractive_index



# ------------
# BUILD SYSTEM
#-------------

# Helper to create a standard test beam
def create_test_beam():
    return GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

def test_build_integration_basic_structure():
    """
    Verifies that calling build() on a valid system produces a runnable Simulation.
    Space(10) -> Lens(50) -> Space(20)
    """
    sys = System(ambient_n=1.0)
    
    # 0. Add Input Beam (Required for valid simulation)
    sys.add_input(create_test_beam())
    
    # 1. Add Elements
    sys.add(Space(d=10.0))
    sys.add(ThinLens(f=50.0))
    sys.add(Space(d=20.0))
    
    # 2. Build
    sim = sys.build()
    
    # 3. Verify Layout Resolution
    assert len(sim.steps) == 3
    
    # 4. Verify Input is passed through
    assert isinstance(sim.sources, GaussianBeam)
    assert sim.sources.wavelength == 1064e-9


def test_build_integration_variables_and_optimization():
    """
    Verifies that variables marked in the system are correctly exposed 
    in the Simulation object's optimization vector.
    """
    sys = System(ambient_n=1.0)
    sys.add_input(create_test_beam())
    
    # Space with Variable Length
    s1 = Space(d=10.0)
    s1.variable('d') # Mark as optimizable
    sys.add(s1)
    
    # Lens with Variable Focal Length
    l1 = ThinLens(f=100.0)
    l1.variable('f')
    sys.add(l1)
    
    sim = sys.build()
    
    # 1. Check Vector Size (2 variables)
    assert len(sim.initial_values) == 2
    
    # 2. Check Values
    vals = sorted(sim.initial_values)
    assert np.allclose(vals, [10.0, 100.0])
    
    # 3. Verify Transform Function (The "Math Kernel")
    new_theta = np.array([15.0, 200.0]) # d=15, f=200
    state_vector = sim.parameter_transform(new_theta)
    
    # The state vector contains all roots (params, L, n).
    assert 15.0 in state_vector
    assert 200.0 in state_vector


def test_build_integration_auto_spacer_wiring():
    """
    Verifies the complex chain: 
    Variable Space -> AutoSpacer (Constraint) -> Fixed Anchor
    """
    sys = System()
    sys.add_input(create_test_beam())
    
    # Variable upstream
    d_param = Parameter(10.0, "d_var", fixed=False)
    sys.add(Space(d=d_param))
    
    # Fixed Anchor at 100
    sys.add(ThinLens(f=50), z=100.0)
    
    sim = sys.build()
    
    # 1. Structure: [Space(Var), AutoSpacer, Lens]
    assert len(sim.steps) == 3
    
    # 2. Verify Simulation behavior
    # Run with initial values (d=10)
    initial_state = sim.parameter_transform(sim.initial_values)
    
    # Locate AutoSpacer length in the state vector
    # Step 1 is the AutoSpacer
    spacer_step = sim.steps[1]
    spacer_L = initial_state[spacer_step.length_index]
    
    # Expect 100 - 10 = 90
    assert spacer_L == 90.0
    
    # 3. Modify Variable and Re-run
    new_theta = np.array([40.0]) 
    new_state = sim.parameter_transform(new_theta)
    
    new_spacer_L = new_state[spacer_step.length_index]
    
    # Expect 100 - 40 = 60
    assert new_spacer_L == 60.0


def test_build_integration_refractive_index_flow():
    """
    Verifies that 'inherit_n' is correctly resolved to actual values 
    in the final simulation steps.
    """
    sys = System(ambient_n=1.33)
    sys.add_input(create_test_beam())
    
    # Space (inherits 1.33)
    sys.add(Space(d=10))
    
    sim = sys.build()
    
    # Get the state
    state = sim.parameter_transform(sim.initial_values)
    
    # Step 0: Space
    step0 = sim.steps[0]
    n_val = state[step0.index_index]
    
    assert n_val == 1.33