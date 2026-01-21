import pytest
import math
from diffractix.graph.ast import Node, Constant, Parameter, Symbol, InputNode, BinaryOp, UnaryOp
import gc 

# --------------------------
# 1. BASIC ARITHMETIC CHECKS
# --------------------------
def test_constant_math():
    """Test standard math operations between Constants."""
    c1 = Constant(10)
    c2 = Constant(5)

    # Addition
    res = c1 + c2
    assert isinstance(res, BinaryOp)
    assert res.value == 15
    assert res.is_constant

    # Multiplication with raw scalar (should auto-convert)
    res = c1 * 2
    assert isinstance(res, BinaryOp)
    assert res.right.value == 2 
    assert res.value == 20

    # Division
    res = c1 / c2
    print(res.value)
    assert res.value == 2.0

def test_parameter_math():
    """Test math with Parameters (Variables)."""
    p = Parameter(value=4.0, name="p", fixed=False)
    
    # Operations
    f = p ** 0.5  # Sqrt
    assert f.value == 2.0
    assert not f.is_constant  # Should inherit 'variable' status from p

    # Negation (UnaryOp)
    neg = -p
    assert isinstance(neg, UnaryOp)
    assert neg.value == -4.0


# -----------------------------------
# 2. INPUT NODE (LEAF-BOXING) TESTS
# -----------------------------------
def test_input_node_transparency():
    """
    Verify InputNode behaves exactly like the node it wraps 
    for math and attribute access.
    """
    p = Parameter(value=10.0, name="test", fixed=True)
    box = InputNode(p)

    # 1. Attribute Forwarding
    assert box.value == 10.0
    assert box.name == "test"
    assert box.fixed is True

    # 2. Attribute Setting (Forwarding to inner node)
    box.fixed = False
    assert p.fixed is False  # Should have modified the inner parameter

    # 3. Math Transparency
    res = box + 5
    assert res.value == 15.0
    # The BinaryOp should hold the BOX, not the parameter
    # This is crucial for the "Hot Swap" to work later
    assert res.left is box or res.right is box


def test_input_node_hot_swapping():
    """
    The Holy Grail Test: 
    Verify we can swap the content of an InputNode and 
    instantly update a pre-existing math formula.
    """
    # 1. Setup Initial State
    param_a = Parameter(10.0, name="A", fixed=True)
    handle_a = InputNode(param_a)
    
    # 2. Build AST Formula:  f = A * 2
    formula = handle_a * 2
    assert formula.value == 20.0

    # 3. HOT SWAP: Change the node inside the handle
    # Simulates: lens.f = 5.0 (replacing Parameter(10) with Parameter(5))
    new_param = Parameter(5.0, name="A", fixed=True)
    handle_a.node = new_param

    # 4. Verify AST automatically updates
    # The 'formula' BinaryOp still holds 'handle_a', which now points to 'new_param'
    assert formula.value == 10.0 


def test_input_node_swap_constant_to_formula():
    """
    Test upgrading a scalar parameter to a complex formula 
    without breaking downstream references.
    """
    # Setup: X = 10
    handle_x = InputNode(Constant(10.0))
    
    # Downstream: Y = X + 5
    y = handle_x + 5
    assert y.value == 15.0
    
    # Swap: X becomes a formula (e.g., Z * 2)
    z = Parameter(3.0, name="Z", fixed=False)
    formula_node = z * 2  # Value is 6.0
    
    # Inject formula into the handle
    handle_x.node = formula_node
    
    # Y should now calculate: (3.0 * 2) + 5 = 11.0
    assert y.value == 11.0
    assert not y.is_constant # Y became variable because Z is variable!


# --------------------------
# 3. SYMBOL & BINDING TESTS
# --------------------------
def test_symbol_caching():
    """Ensure Symbol('x') always returns the same object instance."""
    s1 = Symbol("ambient_n")
    s2 = Symbol("ambient_n")
    s3 = Symbol("other")

    assert s1 is s2         # Identity check
    assert s1 is not s3     # Different names -> different objects
    
    # Ensure cache doesn't prevent new independent symbols
    assert s3.name == "other"


def test_symbol_binding_resolution():
    """
    Test creating a math tree with a Void Symbol, 
    then binding it to values later.
    """
    sym = Symbol("focal_length")
    
    # 1. Use Symbol in Math (Pre-Binding)
    # Result is a BinaryOp holding a Symbol
    formula = sym / 2.0 
    
    # Accessing value before binding should crash
    with pytest.raises(ValueError, match="not been bound"):
        _ = formula.value

    # 2. Bind Symbol to a Parameter
    p = Parameter(100.0, name="f_optim", fixed=False)
    sym.bind(p)  # Bind it!

    # 3. Check Resolution
    assert sym.value == 100.0
    assert formula.value == 50.0  # 100 / 2
    assert not formula.is_constant # Should inherit variable status from p


def test_symbol_leaf_boxing_integration():
    """
    Verify how OpticalElements interact with Symbols via InputNodes.
    System: lens.n = Symbol("ambient")
    """
    # 1. The user creates a Symbol
    sym = Symbol("ambient")
    
    # 2. The Element wraps it in an InputNode (simulated)
    handle = InputNode(sym)
    
    # 3. Math uses the handle
    output = handle * 3.0
    
    # 4. Bind the Symbol (System Build Phase)
    env_param = Constant(1.33)
    sym.bind(env_param)
    
    # 5. Verify Propagation
    # Output -> handle -> sym -> env_param
    assert output.value == 1.33 * 3.0


# --------------------------
# 4. COMPLEX GRAPH INTEGRITY
# --------------------------
def test_complex_graph_structure():
    """Test a multi-level AST with mixed types."""
    # Formula: y = (a + b) * (c - 1)
    a = Parameter(1, "a", fixed=True)
    b = Parameter(2, "b", fixed=True)
    c = Parameter(5, "c", fixed=True)

    # Use InputNodes as standard practice
    ha = InputNode(a)
    hb = InputNode(b)
    hc = InputNode(c)

    term1 = ha + hb  # 3
    term2 = hc - 1   # 4
    y = term1 * term2 # 12

    assert y.value == 12.0
    
    # Check graph structure
    assert isinstance(y, BinaryOp)
    assert y.left is term1 or y.right is term1
    
    # Modifying 'a' should propagate
    ha.node = Constant(2) # a becomes 2
    # term1 -> 2 + 2 = 4
    # y -> 4 * 4 = 16
    assert y.value == 16.0



# --------------------------
# 5. SYMBOL MISUSE TESTS
# --------------------------
def test_symbol_unbound_access_error():
    """
    Misuse: Trying to read the value of a Symbol before it is bound.
    Expected: ValueError.
    """
    sym = Symbol("orphan")
    
    # 1. Direct value access
    with pytest.raises(ValueError, match="not been bound"):
        _ = sym.value

    # 2. Indirect value access via Math
    # The graph construction (sym + 1) works, but evaluation must fail
    expr = sym + 1
    with pytest.raises(ValueError, match="not been bound"):
        _ = expr.value

def test_symbol_unbound_setter_error():
    """
    Misuse: Trying to set the value of a Symbol before it is bound.
    Expected: ValueError (cannot forward assignment to None).
    """
    sym = Symbol("orphan_write")
    
    with pytest.raises(ValueError, match="not been bound"):
        sym.value = 5.0

def test_symbol_binding_consistency():
    """
    Stress: Verify that binding a symbol updates all mathematical 
    references to it immediately.
    """
    sym = Symbol("x")
    
    # Create a chain of math relying on this void symbol
    f1 = sym * 2
    f2 = f1 + 10
    f3 = f2 ** 2
    
    # Late binding
    p = Parameter(5.0, name="bound_val", fixed=True)
    sym.bind(p)
    
    # f1 = 5*2 = 10
    # f2 = 10+10 = 20
    # f3 = 20^2 = 400
    assert f3.value == 400.0


# --------------------------
# 6. INPUT NODE MISUSE
# --------------------------
def test_input_node_attribute_forwarding():
    """
    Check that InputNode correctly masks itself and forwards 
    attributes to the underlying node.
    """
    p = Parameter(10.0, name="p", fixed=True)
    box = InputNode(p)
    
    # Test reading non-existent attribute (should raise AttributeError from Parameter, not InputNode)
    with pytest.raises(AttributeError):
        _ = box.non_existent_attr

    # Test that 'node' attribute is protected/internal
    # Accessing .node should return the Parameter
    assert box.node is p


# --------------------------
# 7. TYPE SAFETY & INVALID MATH
# --------------------------
def test_invalid_type_math():
    """
    Misuse: Trying to add incompatible types (e.g., string) to a Node.
    The AST tries to wrap literals in Constant(), which tries float().
    Expected: ValueError or TypeError during graph construction.
    """
    c = Constant(1)
    
    # Adding a string should fail conversion to float inside Constant.__init__
    with pytest.raises(ValueError):
        res = c + "not_a_number"

def test_none_math_error():
    """
    Misuse: Adding None to a Node.
    Expected: TypeError (float() argument must be a string or a number).
    """
    c = Constant(1)
    with pytest.raises(TypeError):
        res = c + None


# --------------------------
# 8. CYCLIC DEPENDENCY (STRESS)
# --------------------------
def test_circular_dependency_recursion():
    """
    Stress: Create a graph cycle and ensure it raises RecursionError 
    instead of crashing the interpreter or hanging.
    
    Scenario: A = B + 1, B = A + 1
    This is hard to do with immutable Nodes, but possible via InputNode hot-swapping.
    """
    # 1. Setup handles
    pA = Parameter(0, "A", fixed=False)
    pB = Parameter(0, "B", fixed=False)
    
    handle_A = InputNode(pA)
    handle_B = InputNode(pB)
    
    # 2. Create crossed wires
    # A becomes (B + 1)
    node_A_new = handle_B + 1
    # B becomes (A + 1)
    node_B_new = handle_A + 1
    
    # 3. Apply the cycle via hot-swap
    handle_A.node = node_A_new
    handle_B.node = node_B_new
    
    # 4. Evaluate
    # Accessing handle_A.value calls node_A_new.value -> handle_B.value -> node_B_new.value -> handle_A.value ...
    with pytest.raises(RecursionError):
        _ = handle_A.value


# --------------------------
# 9. DEEP GRAPH (PERFORMANCE)
# --------------------------
def test_deep_recursion_limit():
    """
    Stress: Evaluate a very deep AST to check Python's recursion limit.
    """
    # Create a chain: x + 1 + 1 + 1 ...
    root = Constant(0)
    curr = root
    
    depth = 500  # Safe limit (default python limit is usually 1000)
    for _ in range(depth):
        curr = curr + 1
        
    assert curr.value == depth
    
    # Test that we didn't break repr recursion either
    assert isinstance(repr(curr), str)


# --------------------------
# 10. BASIC DEDUPLICATION
# --------------------------
def test_binary_op_caching():
    """Verify that identical math operations return the same instance."""
    c = Constant(10)
    
    # Create two separate operations that are mathematically identical
    op1 = c + 5
    op2 = c + 5
    
    assert op1 is op2, "Cache failed: Identical BinaryOps should be deduped"

def test_commutativity_caching():
    """
    Verify that a + b returns the same object as b + a.
    (Your BinaryOp.__hash__ implementation explicitly handles this sort order)
    """
    a = Parameter(1, "a", fixed=True)
    b = Parameter(2, "b", fixed=True)
    
    op1 = a + b
    op2 = b + a
    
    assert op1 is op2, "Cache failed: Commutative operations should be normalized"


def test_symbol_aliasing_behavior():
    """
    Verifies that symbols with the same name are the same object 
    and that rebinding one affects the other immediately.
    """
    # 1. Setup
    x = Parameter(value=10.0, name="x", fixed=True)
    y = Parameter(value=10.0, name="y", fixed=True)

    # 2. Define Aliased Symbols
    # Because the name "a" is the same, the cache returns the SAME instance.
    a = Symbol("a")
    b = Symbol("a")
    
    assert a is b, "Cache failed: Symbols with identical names must be the same object"

    # 3. Define Math
    z = x + y * x  # 10 + 100 = 110
    w = y / x      # 10 / 10 = 1.0

    c = a + b      # Symbol("a") + Symbol("a")

    # 4. Bind 'a' -> 110
    # c is now effectively 110 + 110 = 220
    a.bind(z)
    assert c.value == 220.0

    # 5. Bind 'b' -> 1.0
    # Since b IS a, this overwrites the previous binding.
    # Symbol("a") now points to 1.0
    b.bind(w)

    # 6. Verify Equivalence
    # (w + w) = 1.0 + 1.0 = 2.0
    # c       = 1.0 + 1.0 = 2.0
    print(f"\n(w + w).value = {(w + w).value}")
    print(f"c.value       = {c.value}")

    assert (w + w).value == c.value
    assert c.value == 2.0



# --------------------------
# 11. SYMBOL LIFECYCLE & EDGE CASES
# --------------------------

def test_symbol_self_binding_cycle():
    """
    Edge Case: User binds a symbol to an expression containing itself.
    Formula: x = x + 1
    Expected: RecursionError (Infinite loop detection via Python stack depth)
    """
    s = Symbol("infinite")
    
    # Create expression: s + 1
    expr = s + 1
    
    # Bind s -> (s + 1)
    s.bind(expr)
    
    # Evaluation should spiral infinitely
    with pytest.raises(RecursionError):
        _ = s.value

def test_distinct_symbols_independence():
    """
    Verify that symbols with DIFFERENT names are distinct objects,
    even if they are bound to identical values.
    """
    s1 = Symbol("alpha")
    s2 = Symbol("beta")
    
    assert s1 is not s2, "Symbols with different names must be unique instances"
    
    # Bind to same value
    s1.bind(10.0)
    s2.bind(10.0)
    
    assert s1.value == s2.value
    
    # Rebind s1, s2 should remain unchanged
    s1.bind(20.0)
    assert s1.value == 20.0
    assert s2.value == 10.0

def test_symbol_garbage_collection():
    """
    Stress: Verify that Symbols are correctly evicted from the cache 
    when no longer referenced.
    """
    # 1. Create a symbol in a local scope
    name = "transient_symbol"
    s = Symbol(name)
    
    # Get the cache key
    key = s.canonical_key()
    
    # It must be in the cache
    assert key in Node._cache
    
    # 2. Delete the strong reference
    del s
    
    # 3. Force Garbage Collection
    gc.collect()
    
    # 4. Verify Eviction
    # If this fails, you have a memory leak (Node._cache is holding a strong ref somewhere)
    assert key not in Node._cache
