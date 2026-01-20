import pytest
from diffractix.graph.ast import Node, Constant, Parameter, Symbol, InputNode, BinaryOp
from diffractix.elements import ThinLens, Space, ABCD

# --------------------------
# 1. STRUCTURE CHECKS
# --------------------------
def test_element_wraps_params_in_input_node():
    """
    Verify that initializing an element automatically wraps 
    scalars in InputNode(Parameter(...)).
    """
    lens = ThinLens(f=100.0)
    
    # 1. The public attribute should be an InputNode (The Handle)
    assert isinstance(lens.f, InputNode)
    
    # 2. The internal node should be a Parameter (The Body)
    assert isinstance(lens.f.node, Parameter)
    assert lens.f.node.value == 100.0
    assert lens.f.node.fixed is True # Defaults to fixed


def test_element_accepts_raw_nodes():
    """
    Verify we can initialize an element with an existing Node 
    (Parameter, Constant, or Symbol) and it gets boxed correctly.
    """
    # Case A: Parameter
    p = Parameter(50.0, "external_param", fixed=False)
    lens = ThinLens(f=p)
    
    assert isinstance(lens.f, InputNode)
    assert lens.f.node is p  # Should hold the exact instance
    assert lens.f.value == 50.0

    # Case B: Symbol
    s = Symbol("sys_f")
    lens_sym = ThinLens(f=s)
    
    assert isinstance(lens_sym.f, InputNode)
    assert lens_sym.f.node is s


# --------------------------
# 2. INTER-ELEMENT MATH
# --------------------------
def test_linking_elements_via_formula():
    """
    Scenario: Lens2's focal length is defined as half of Lens1's.
    L2.f = L1.f / 2
    """
    l1 = ThinLens(f=100.0, label="L1")
    l2 = ThinLens(f=50.0, label="L2") # Initial value doesn't matter
    
    # Define dependency
    # Note: l1.f is an InputNode. Math on InputNode returns a BinaryOp.
    l2.f = l1.f / 2.0 
    
    # Check Structure
    # l2.f is an InputNode -> wrapping a BinaryOp -> (l1.f / 2)
    assert isinstance(l2.f, InputNode)
    assert isinstance(l2.f.node, BinaryOp)
    assert l2.f.value == 50.0
    
    # Check Dependency (Hot Swap L1)
    l1.f = 200.0
    # L2 should automatically update
    assert l2.f.value == 100.0


# --------------------------
# 3. SYMBOLIC INTEGRATION
# --------------------------
def test_element_symbol_resolution():
    """
    Test the flow: Symbol -> Element -> Math -> Binding -> Value
    """
    # 1. Create Symbol
    n_env = Symbol("n_environment")
    
    # 2. Assign to Element (Space)
    # Space(d=10, n=Symbol)
    s = Space(d=10.0, n=n_env)
    
    # 3. Use Element property in a formula
    # Optical Path Length = d * n
    opl = s.d * s.n 
    
    # 4. Verify crash before binding
    with pytest.raises(ValueError, match="bound"):
        _ = opl.value
        
    # 5. Bind Symbol
    env_param = Constant(1.5)
    n_env.target = env_param
    
    # 6. Verify Values
    assert s.n.value == 1.5
    assert opl.value == 15.0 # 10 * 1.5


# --------------------------
# 4. HOT-SWAPPING & AST STABILITY
# --------------------------
def test_hot_swap_preserves_external_graph():
    """
    Verify that changing an Element's parameter updates 
    external graphs that rely on it.
    """
    # Setup
    lens = ThinLens(f=10.0)
    
    # External Graph (e.g., total power calculation)
    # Power = 1/f
    power_ast = 1.0 / lens.f
    
    assert power_ast.value == 0.1
    
    # SWAP 1: Change to a new scalar
    lens.f = 5.0
    assert power_ast.value == 0.2  # Updated!
    
    # SWAP 2: Change to a Formula (BinaryOp)
    # f = 2 * x
    x = Parameter(2.0, "x", fixed=False)
    lens.f = x * 2.0 # f is now 4.0
    
    assert lens.f.value == 4.0
    assert power_ast.value == 0.25 # 1 / 4.0
    
    # SWAP 3: Update the variable 'x'
    # This checks depth: power_ast -> lens.f(InputNode) -> BinaryOp -> x
    # This is 3 levels of indirection!
    x.value = 4.0 # f becomes 8.0
    assert power_ast.value == 0.125

