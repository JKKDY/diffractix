import pytest
from diffractix.graph.ast import Node, Constant, Parameter, Symbol, InputNode, BinaryOp
from diffractix.elements import ThinLens, Space, ABCD

# -------------------
# 1. STRUCTURE CHECKS
# -------------------
def test_element_wraps_params_in_input_node():
    """
    Verify that initializing an element automatically wraps 
    scalars in InputNode(Parameter(...)).
    """
    lens = ThinLens(f=100.0)
    
    # The public attribute should be an InputNode (The Handle)
    assert isinstance(lens.f, InputNode)
    
    # The internal node should be a Parameter (The Body)
    assert isinstance(lens.f.node, Parameter)
    assert lens.f.node.value == 100.0
    assert lens.f.node.fixed is True # Defaults to fixed

    # also test for ABCD element
    abcd = ABCD(A=1.0, B=2.0, C=3.0, D=4.0)
    
    assert isinstance(abcd.A, InputNode)
    assert isinstance(abcd.B, InputNode)
    assert isinstance(abcd.C, InputNode)
    assert isinstance(abcd.D, InputNode)
    assert isinstance(abcd.n, InputNode)
    assert isinstance(abcd.thickness, InputNode)
    
    assert isinstance(lens.f.node, Parameter)
    assert abcd.A.node.value == 1.0
    assert abcd.B.node.value == 2.0
    assert abcd.C.node.value == 3.0
    assert abcd.D.node.value == 4.0
    assert abcd.A.value == 1.0
    assert abcd.B.value == 2.0
    assert abcd.C.value == 3.0
    assert abcd.D.value == 4.0


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


def test_element_axcepts_nodes_and_floats_but_not_other_types():
    """
    Verify that initializing an element with invalid types raises errors.
    Valid types: float, Parameter, Symbol
    Invalid types: str, list, dict, etc.
    """
    a = ThinLens(f=100.0)  
    b = ThinLens(f=Symbol("x")) 
    c = ThinLens(f=Parameter(10, name="param", fixed=False))
    d = ThinLens(f=a.f+b.f)  
    e = ThinLens(f=None)

    with pytest.raises(TypeError):
        _ = ThinLens(f="not_a_number")
    
    with pytest.raises(TypeError):
        _ = ThinLens(f=[1.0, 2.0])
    
    with pytest.raises(TypeError):
        _ = ThinLens(f={"A": 1.0})


# ---------------------
# 2. INTER-ELEMENT MATH
# ---------------------
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


# -----------------------
# 3. SYMBOLIC INTEGRATION
# -----------------------
def test_element_symbol_resolution():
    """
    Test the flow: Symbol -> Element -> Math -> Binding -> Value
    """
    # Create Symbol
    n_env = Symbol("n_environment")
    
    # Assign to Element (Space)
    # Space(d=10, n=Symbol)
    s = Space(d=10.0, n=n_env)
    
    # Use Element property in a formula
    # Optical Path Length = d * n
    opl = s.d * s.n 
    
    # Verify crash before binding
    with pytest.raises(ValueError, match="bound"):
        _ = opl.value
        
    # Bind Symbol
    env_param = Parameter(1.5, fixed=False, name="env_n")
    n_env.bind(env_param)
    
    # verify
    assert s.n.value == 1.5
    assert opl.value == 15.0 # 10 * 1.5
    assert opl.is_constant is False # Because n is variable

    # now rebind
    n_env.bind(2)
    assert s.n.value == 2
    assert opl.value == 20.0 # 10 * 2
    assert opl.is_constant is True # Because n is variable


# -------------------------------
# 4. HOT-SWAPPING & AST STABILITY
# -------------------------------
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
    assert lens.f.is_constant is False
    assert power_ast.value == 0.25 # 1 / 4.0
    
    # SWAP 3: Update the variable 'x'
    # This checks depth: power_ast -> lens.f(InputNode) -> BinaryOp -> x
    # This is 3 levels of indirection!
    x.value = 4.0 # f becomes 8.0
    assert power_ast.value == 0.125


# --------------------------------
# 5. COMPILER/OPTIMIZER VISIBILITY
# --------------------------------
def test_constness_propagation():
    """
    Verify that 'is_constant' (fixed) status propagates correctly 
    through the graph. This is critical for the Compiler to know 
    which parameters to add to DOF (degrees of freedom) vector.
    """
    # Case 1: All Fixed
    p1 = Parameter(10.0, "p1", fixed=True)
    p2 = Parameter(20.0, "p2", fixed=True)
    
    # Operations on fixed nodes should be constant
    formula_fixed = p1 + p2
    assert formula_fixed.is_constant is True
    
    # Case 2: One Variable
    p_var = Parameter(5.0, "p_var", fixed=False)
    
    # Operation involving a variable should NOT be constant
    formula_var = p1 + p_var
    assert formula_var.is_constant is False
    
    # Case 3: Element Wrapper Propagation
    lens = ThinLens(f=p_var)
    # The InputNode wrapper must report the inner node's status
    assert lens.f.is_constant is False
    assert lens.f.fixed is False # InputNode forwards .fixed attribute
    
    # Case 4: Deep Propagation
    # Space.d depends on Lens.f (variable)
    space = Space(d=lens.f * 2, n=1.0)
    assert space.d.is_constant is False
    
    # If we fix the source parameter, the whole chain should become fixed
    p_var.fixed = True
    assert lens.f.is_constant is True
    assert space.d.is_constant is True



def symbol_rebinding():
    """
    Verify that initializing an element with invalid types raises errors.
    Valid types: float, Parameter, Symbol
    Invalid types: str, list, dict, etc.
    """
    a = ThinLens(f=100.0)  
    b = ThinLens(f=Symbol("x")) 
    c = ThinLens(f=Parameter(10, name="param", fixed=False))
    d = ThinLens(f=a.f+b.f)  
    # e = ThinLens(f=None)

    x = Symbol("x")
    x.bind(5.0)

    assert b.f.value == 5.0

    x.bind(10.0)
    assert b.f.value == 10.0


def test_rebinding(): 
    l1 = ThinLens(f=100.0)
    l2 = ThinLens(f=50.0)
    s = Space(d = l1.f + l2.f)

    assert s.d.value == 150.0

    l1.f = 200.0
    assert s.d.value == 250.0

    l2.f = l1.f
    assert s.d.value == 400.0 # Because now l2.f is also 200.0


    x = Parameter(10.0, "x", fixed=False)
    x = x + x
    l1.f = x

    assert l1.f.value == 20.0
    assert s.d.value == 40 

    # 2. test
    b = ThinLens(f=Symbol("x")) 
    x = Symbol("x")
    x.bind(5.0)

    assert b.f.value == 5.0

    x.bind(10.0)

    assert b.f.value == 10.0