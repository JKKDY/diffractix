import pytest
import numpy as np
from diffractix.graph.ast import Parameter, InputNode, Symbol, Constant
from diffractix.graph.compile import collect_leaves, compile_parameter_transform

# ---------------------
# LEAF COLLECTION TESTS
# ---------------------
# leaves are the variable parameters that the compiler needs to optimize over. 
# i.e. the degrees of freedom.
def test_collect_leaves_basic():
    """
    Test filtering of fixed vs variable parameters.
    """
    p_var = Parameter(1.0, "var", fixed=False)
    p_fix = Parameter(2.0, "fix", fixed=True)
    
    # Roots can be the parameters themselves or nodes containing them
    roots = [p_var, p_fix]
    
    leaves = collect_leaves(roots)
    
    # Should only contain the variable parameter
    assert len(leaves) == 1
    assert leaves[0] is p_var

def test_collect_leaves_deduplication():
    """
    Test that the same parameter used in multiple places 
    only appears ONCE in the variable vector.
    """
    p = Parameter(10.0, "shared", fixed=False)
    
    # A graph where 'p' is used twice:
    # 1. Directly
    # 2. Inside a formula (p * 2)
    root1 = p
    root2 = p * 2.0 
    
    leaves = collect_leaves([root1, root2])
    
    # p must not appear twice
    assert len(leaves) == 1
    assert leaves[0] is p

def test_collect_leaves_through_indirection():
    """
    Test traversal through InputNodes and Symbols.
    """
    # Setup: Param -> Symbol -> InputNode
    p = Parameter(5.0, "source", fixed=False)
    sym = Symbol("alias")
    sym.bind(p)
    
    handle = InputNode(sym)
    
    leaves = collect_leaves([handle])
    
    assert len(leaves) == 1
    assert leaves[0] is p

def test_collect_leaves_unbound_symbol_error():
    """
    Compiler should crash if it hits an unbound symbol.
    """
    sym = Symbol("orphan")
    # No binding
    
    with pytest.raises(ValueError, match="unbound"):
        collect_leaves([sym])


# ------------------------
# TRANSFORM FUNCTION TESTS
# ------------------------
# transform function maps theta (degrees of freedom vector) to the 
def test_parameter_transform_execution():
    """
    Verify that the generated P(theta) function (theta = degrees of freedom vector) functions correctly:
    1. Updates the parameter values.
    2. Returns the correctly evaluated roots.
    """
    # Graph: y = m*x + c
    m = Parameter(2.0, "m", fixed=False) # Variable
    x = Parameter(3.0, "x", fixed=False) # Variable
    c = Parameter(1.0, "c", fixed=True)  # Fixed
    
    # Root node (the calculation)
    y = m * x + c
    
    # 1. Generate Transform
    # Note: We pass [y] as the root we want to evaluate
    transform_func, initial_theta, vars_list = compile_parameter_transform([y])
    
    # Check initial state extraction
    # Order depends on traversal, usually [m, x] or [x, m] depending on binary op internals
    assert len(initial_theta) == 2
    assert vars_list == [m, x] # (Assuming left-first traversal)
    np.testing.assert_array_equal(initial_theta, [2.0, 3.0])
    
    # 2. Run Transform with NEW values
    # Let's change m -> 10, x -> 5. (c stays 1)
    # Expected y = 10 * 5 + 1 = 51
    new_theta = np.array([10.0, 5.0])
    
    result = transform_func(new_theta)
    
    # Check Output (The evaluated root)
    assert result[0] == 51.0
    


def test_transform_ordering_consistency():
    """
    Ensure the order of theta aligns with the order of variable_params.
    """
    p1 = Parameter(1.0, "p1", fixed=False)
    p2 = Parameter(2.0, "p2", fixed=False)
    
    roots = [p1, p2]
    
    transform_func, _, vars_list = compile_parameter_transform(roots)
    
    # Determine the compiled order
    assert vars_list == [p1, p2]
    
    # Pass distinct values [10, 20]
    # If aligned: p1 gets 10, p2 gets 20. 
    # If flipped: p1 gets 20, p2 gets 10.
    res = transform_func(np.array([10.0, 20.0]))
    
    
    # The result should also match roots order
    assert res[0] == 10.0
    assert res[1] == 20.0