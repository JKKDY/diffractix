import autograd.numpy as np
from typing import List, Tuple, Set, Dict, Callable
from .ast import Node, Parameter, InputNode, Symbol



def collect_leaves(roots: List[Node]) -> list[Node]:
    """
    Traverses the graph to find unique variable Parameters by finding all leaves of the AST
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


def generate_parameter_transform(roots: List[Node]) -> Tuple[Callable, np.ndarray, List[Parameter]]:
    
    # variable vector (len = degrees of freedom)
    variable_params = collect_leaves(roots) 

    # initial values
    initial_values = np.array([p.value for p in variable_params], dtype=float)

    def transform(theta_in: np.ndarray) -> np.ndarray:
        # Update the Source Parameters
        # This injects the new values (floats or Tracers) into the graph leaves.
        # Since the graph is connected by references, this updates the whole state.
        for param, new_val in zip(variable_params, theta_in):
            param.value = new_val
            
        # evaluate 
        return np.array([r.value for r in roots])

    return transform, initial_values, variable_params

