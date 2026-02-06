import autograd.numpy as np
from typing import List, Tuple, Set, Dict, Callable
from .ast import Node, Parameter, InputNode, Symbol, BinaryOp, UnaryOp, Constant



def collect_leaves(roots: List[Node]) -> list[Node]:
    """
    Traverses the graph to find unique variable Parameters by finding all (variable) leaves of the AST

    In the context of diffractix the roots are dependent variables (e.g. y in y = x + z) 
    The leaves are independent variables i.e. the degrees of freedom of the system 
    Note that a variable can be a root as well as a leaf (e.g. a simple input value like Lens.f = 1.0 -> f depends only on "itself") 
    We also only consider the "variable" leaves i.e. those whose values can be changed (Constants & fixed Parameters are not part of this)
    """
    seen = set()
    leaves = []

    def collect_leaves_impl(node: Node):
        if node in seen: return
        else: seen.add(node)

        # source paremter -> a potential leaf
        if isinstance(node, Parameter):
            if not node.fixed:
                leaves.append(node)
            return  # If fixed, we don't add to theta, but we stop traversing.

        # symbol -> potential leaf but must be bound to a value
        if isinstance(node, Symbol):
            if node._target is None: 
                raise ValueError(f"Symbol {node} is unbounded.")
            collect_leaves_impl(node._target)
            return

        # its an input node -> handle indirection
        if isinstance(node, InputNode):
            collect_leaves_impl(node.node)
            return

        # its an operation (Binary/Unary)
        if hasattr(node, 'left'): collect_leaves_impl(node.left)
        if hasattr(node, 'right'): collect_leaves_impl(node.right)
        if hasattr(node, 'operand'): collect_leaves_impl(node.operand)
    
    for root in roots:
        collect_leaves_impl(root)

    return leaves


def compile_parameter_transform(roots: List[Node]) -> Tuple[Callable, np.ndarray, List[Parameter]]:
    """
    Compiles the computational graph into a differentiable function P(theta) where theta is the degrees of freedom vector

    Args:
        roots: A list of AST Nodes (Sinks) that represent the required outputs of the 
               simulation (e.g., matrix elements, lengths, refractive indices).

    Returns:
        transform: A function f(theta) -> values. 
                   - Input: 1D array of parameter values (floats or autograd tracers).
                   - Output: 1D array corresponding to the evaluated values of 'roots'.
                   - Side Effect: Updates the .value attribute of the underlying Parameter objects.
        initial_theta: A numpy array containing the starting values of the variable parameters.
        variable_params: The ordered list of Parameter objects corresponding to 'initial_theta'.
    """
    # variable vector (len = degrees of freedom)
    variable_params = collect_leaves(roots) 

    # initial values
    initial_values = np.array([p.value for p in variable_params], dtype=float)

    def transform(theta_in: np.ndarray) -> np.ndarray:
        assert len(theta_in) == len(variable_params) 
      
        # memoization cache to ensure O(N) and prevent re-evaluating shared branches
        memo =  {p: v for p, v in zip(variable_params, theta_in)}

        def eval_node(node: Node):
            # check cachse
            if node in memo: 
                return memo[node]
            
            # compute 
            if isinstance(node, Parameter):
                # If it's variable, get from theta. If fixed, use existing .value
                val = memo.get(node, node.value)
            
            elif isinstance(node, BinaryOp):
                val = node.op.func(eval_node(node.left), eval_node(node.right))
                
            elif isinstance(node, UnaryOp):
                val = node.op.func(eval_node(node.operand))
                
            elif isinstance(node, Constant):
                val = node.value

            elif isinstance(node, (InputNode, Symbol)):
                # Indirection: Dive deeper
                target = node.node if isinstance(node, InputNode) else node._target
                val = eval_node(target)
            
            # store & return
            memo[node] = val
            return val

        # dispatch evaluation of all roots
        return np.array([eval_node(r) for r in roots])

    return transform, initial_values, variable_params

