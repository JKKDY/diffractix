import pytest
import numpy as np
import autograd.numpy as anp  # For consistency in checks
from dataclasses import dataclass
from diffractix.elements import ThinLens, Space, Mirror, Interface, ABCD
from diffractix.graph import Parameter, InputNode, Symbol

def assert_matrix_close(mat, expected, rtol=1e-5):
    """Helper to check if two 2x2 matrices are approximately equal."""
    assert mat.shape == (2, 2)
    np.testing.assert_allclose(mat, expected, rtol=rtol)


# ---------
# THIN LENS
# ---------
def test_thin_lens_init():
    """Test thin lens instantiation, parameter wrapping, and mutability."""
    # test init
    lens = ThinLens(f=0.1, label="Objective")
    assert isinstance(lens.f, InputNode)
    assert isinstance(lens.f.node, Parameter)
    assert lens.f.value == 0.1
    assert lens.label == "Objective"
    
    # check metadata
    assert lens.f.fixed
    assert lens.length_param_names == [] # Lenses are thin
    assert lens.param_names == ["f"]
    assert lens.variable_parameter_names == []

    # test mutability
    lens.f = 0.2
    assert isinstance(lens.f, InputNode)
    assert isinstance(lens.f.node, Parameter)
    assert lens.f.value == 0.2

    # test variable toggling
    lens.variable()
    assert not lens.f.fixed
    assert lens.variable_parameter_names == ["f"]

    # test fixed toggling
    lens.fixed()
    assert lens.f.fixed
    assert lens.variable_parameter_names == []

    

def test_thin_lens_get_sim_data():
    """Test propagation physics through thin lens."""
    for f in [-0.5, 0.1, 1, np.inf]: 
        lens = ThinLens(f=f)
        
        mat_func, len_func, index_func = lens.get_sim_functions()
        vals = lens.values
        
        # Check args unpacking
        assert vals==[f]
        assert index_func is None

        mat = mat_func(*vals)
        length = len_func(*vals)

        assert length == 0

        expected = np.array([
            [1.0,      0.0],
            [-1.0/f,   1.0]
        ])
        assert_matrix_close(mat, expected)




# ----------
# SPACE TEST
# ----------
def test_space_init():
    """Test Space instantiation, parameter wrapping, and mutability."""
    # 1. Init
    s = Space(d=10.0, n=1.5, label="GlassBlock")
    
    assert isinstance(s.d, InputNode)
    assert isinstance(s.n, InputNode)
    assert isinstance(s.d.node, Parameter)
    assert isinstance(s.n.node, Parameter)
    assert s.d.value == 10.0
    assert s.n.value == 1.5
    assert s.label == "GlassBlock"
    
    # 2. Metadata
    assert s.length_param_names == ["d"]
    assert s.param_names == ["d", "n"]
    assert s.d.fixed and s.n.fixed
    assert s.variable_parameter_names == []

    # test mutability
    s.d = 20.0
    assert isinstance(s.d, InputNode)
    assert isinstance(s.d.node, Parameter)
    assert s.d.value == 20.0

    # test variable toggling
    s.variable('d') # Only make d variable
    assert not s.d.fixed
    assert s.n.fixed
    assert s.variable_parameter_names == ["d"]
    assert s.has_variable_length

    # test fixed toggling
    s.fixed() 
    assert s.d.fixed
    assert s.n.fixed
    assert s.variable_parameter_names == []


def test_space_get_sim_data():
    """Test propagation physics through Space."""
    distances = [0.1, 1.0, 10.0]
    indices = [1.0, 1.5]
    
    for d_val in distances:
        for n_val in indices:
            s = Space(d=d_val, n=n_val)
            
            mat_func, len_func, idx_func = s.get_sim_functions()
            vals = s.values
            
            # Check args unpacking
            assert vals == [d_val, n_val]
            
            # Check Physics
            mat = mat_func(*vals)
            length = len_func(*vals)
            idx = idx_func(*vals)

            assert length == d_val
            assert idx == n_val # Space doesn't change index, just carries it

            # Matrix: [[1, d/n], [0, 1]]
            expected = np.array([
                [1.0, d_val / n_val],
                [0.0, 1.0]
            ])
            assert_matrix_close(mat, expected)




# -----------
# MIRROR TEST
# -----------
def test_mirror_init():
    """Test Mirror instantiation and properties."""
     # test init
    m = Mirror(R=0.5, label="M1")
    assert isinstance(m.R, InputNode)
    assert isinstance(m.R.node, Parameter)

    assert m.R.value == 0.5
    
    # check metadata
    assert m.length_param_names == [] # Mirrors are thin
    assert m.param_names == ["R"]
    assert m.variable_parameter_names == []

    # test mutability
    m.R = np.inf # Make it flat
    assert m.R.value == np.inf
    
    # test variable toggling
    m.variable()
    assert not m.R.fixed
    assert m.variable_parameter_names == ["R"]

    # test fixed toggling
    m.fixed()
    assert m.R.fixed
    assert m.variable_parameter_names == []


def test_mirror_get_sim_data():
    """Test reflection physics for Mirrors."""
    # Test Curved (Concave/Convex) and Flat
    radii = [0.5, -0.5, np.inf]
    
    for R_val in radii:
        m = Mirror(R=R_val)
        
        mat_func, len_func, idx_func = m.get_sim_functions()
        vals = m.values
        
        assert vals == [R_val]
        assert len_func(*vals) == 0.0
        assert idx_func is None # Mirrors don't manage index explicitly in this model

        mat = mat_func(*vals)
        
        # Power calculation check
        if np.isinf(R_val):
            power = 0.0
        else:
            power = -2.0 / R_val
            
        expected = np.array([
            [1.0,   0.0],
            [power, 1.0]
        ])
        assert_matrix_close(mat, expected)


# --------------
# INTERFACE TEST
# --------------
def test_interface_init():
    """Test Interface instantiation (Snell's Law element)."""
    # test init
    iface = Interface(n1=1.0, n2=1.5, R=0.1, label="FrontSurface")
    
    assert isinstance(iface.n1, InputNode)
    assert isinstance(iface.n2, InputNode)
    assert isinstance(iface.R, InputNode)
    assert isinstance(iface.n1.node, Parameter)
    assert isinstance(iface.n2.node, Parameter)
    assert isinstance(iface.R.node, Parameter)
    
    # check metadata
    assert iface.length_param_names == []
    assert set(iface.param_names) == {"n1", "n2", "R"}

    # test mutability
    iface.n2 = 1.6
    assert iface.n2.value == 1.6
    
    # test variable toggling
    iface.variable('R', 'n1') 
    vparams = iface.variable_parameter_names
    assert not iface.R.fixed
    assert not iface.n1.fixed
    assert iface.n2.fixed
    assert 'R' in vparams and 'n1' in vparams

    # test fixed toggling
    iface.fixed() 
    assert iface.R.fixed
    assert iface.n1.fixed
    assert iface.n2.fixed
    assert iface.variable_parameter_names == []


def test_interface_get_sim_data():
    """Test refraction physics (Power & Index change)."""
    n1_val = 1.0
    n2_val = 1.5
    R_val = 0.05 # 5cm
    
    iface = Interface(n1=n1_val, n2=n2_val, R=R_val)
    
    mat_func, len_func, idx_func = iface.get_sim_functions()
    vals = iface.values
    
    # Check Index Update Function
    # (Interface MUST update the simulation index to n2)
    current_idx = idx_func(*vals)
    assert current_idx == n2_val
    
    # Check Matrix (Curved Dielectric Interface)
    mat = mat_func(*vals)
    power = (n1_val - n2_val) / (R_val * n2_val)
    
    expected = np.array([
        [1.0, 0.0],
        [power, n1_val / n2_val]
    ])
    assert_matrix_close(mat, expected)



# ------------------
# ABCD/BLACKBOX TEST
# ------------------
def test_abcd_init():
    """Test generic ABCD element initialization."""
    # 1. Init (Scalar mode)
    el = ABCD(A=2, D=0.5, thickness=0.1, n=1.5, label="Relay")
    
    assert isinstance(el.A, InputNode)
    assert isinstance(el.thickness, InputNode)
    assert isinstance(el.A.node, Parameter)
    assert isinstance(el.thickness.node, Parameter)
    assert el.A.value == 2.0
    assert el.thickness.value == 0.1
    
    # 2. Metadata
    assert el.length_param_names == ["thickness"]
    # param_names for ABCD are typically [A, B, C, D, thickness, n]
    # Check specific implementation if order varies, but set equality is safe
    assert set(el.param_names) == {"A", "B", "C", "D", "thickness", "n"}

    # 3. Mutability
    el.B = 0.1
    assert el.B.value == 0.1
    
    # 4. Variable toggling
    el.variable('thickness')
    assert not el.thickness.fixed
    assert el.A.fixed
    assert el.variable_parameter_names == ["thickness"]


def test_abcd_init_matrix_override():
    """Test initializing via the 'matrix' argument."""
    mat = np.array([[2.0, 0.5], [0.1, 1.0]])
    el = ABCD(matrix_val=mat, thickness=0.1)
    
    assert el.A.value == 2.0
    assert el.B.value == 0.5
    assert el.thickness.value == 0.1
    np.testing.assert_array_equal(el.matrix, mat)


def test_abcd_inheritance_mode():
    """
    Scenario: n=None.
    The element should have 'n' as a parameter (value None),
    and return None for the index function (signaling inheritance).
    """
    el = ABCD(A=1, B=0, C=0, D=1, thickness=1.0, n=None)
    
    # 1. Verify Parameter Structure
    # Static signature of get_matrix is (A, B, C, D, thickness, n) -> 6 params
    assert len(el.parameters) == 6
    assert el.n is None

    # 2. Verify Simulation Wiring
    mat_func, len_func, idx_func = el.get_sim_functions()
    
    # Args that the system will extract from el.parameters
    # Since el.n is None, the last argument passed to functions will be None
    args = [1.0, 0.0, 0.0, 1.0, 1.0, None]
    
    # Matrix Function Check: Must handle n=None safely
    mat = mat_func(*args)
    assert np.allclose(mat, np.eye(2))
    
    # Index Function Check: Must be None to trigger System Build inheritance logic
    assert idx_func is None


def test_abcd_explicit_mode():
    """
    Scenario: n=1.5.
    The element should return a function that returns 'n'.
    """
    el = ABCD(A=1, B=0, C=0, D=1, thickness=1.0, n=1.5)
    
    # 1. Verify Parameter Structure
    assert isinstance(el.n, InputNode)
    assert el.n.value == 1.5
    
    # 2. Verify Simulation Wiring
    mat_func, len_func, idx_func = el.get_sim_functions()
    
    args = [1.0, 0.0, 0.0, 1.0, 1.0, 1.5]
    
    # Index Function Check
    # Your ABCD code returns: lambda a,b,c,d,t,n: n
    # It receives 6 arguments.
    assert idx_func is not None
    assert idx_func(*args) == 1.5


def test_abcd_symbolic_n():
    """Test binding 'n' to a symbol."""
    sym = Symbol("glass_n")
    el = ABCD(n=sym)
    
    # Bind Symbol
    sym.bind(1.5)
    
    # Check physics access
    _, _, idx_func = el.get_sim_functions()
    
    # Args: ..., n=1.5
    args = [1,0,0,1,0, 1.5] 
    
    assert idx_func(*args) == 1.5