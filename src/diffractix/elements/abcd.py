"""
Defines the ABCD 'Black Box' element.
"""
from dataclasses import dataclass, field, InitVar
import autograd.numpy as np
from .base import OpticalElement


@dataclass(kw_only=True)
class ABCD(OpticalElement):
    """
    A black-box optical element defined manually by its matrix.
    Useful for subsystems, unknown optics, or purely mathematical transformations.

    Usage:
        el = ABCD(matrix=np.eye(2))           # Option 1: Full Matrix
        el = ABCD(A=1, B=0.1, C=0, D=1)       # Option 2: Scalars
        el = ABCD()                           # Option 3: Default Identity
    
    Parameters:
        matrix (np.ndarray): The 2x2 ABCD matrix. Overrides A, B, C, D if provided.
        thickness (float): Physical length added to the layout (default: 0.0).
        A, B, C, D (float): Individual matrix components (optional helpers).
    """
    matrix: np.ndarray = field(default=None)
    thickness: float = 0.0  # Physical length added to the layout

    # Helper init-vars to allow ABCD(A=1, ...) syntax
    A: InitVar[float] = None
    B: InitVar[float] = None
    C: InitVar[float] = None
    D: InitVar[float] = None

    @property
    def length_param_names(self):
        """Explicitly links physical length to the 'thickness' parameter."""
        return ["thickness"]

    def __post_init__(self, A, B, C, D):
        # If matrix is explicitly provided, use it (ignoring scalar inputs)
        if self.matrix is not None:
            return

        # If matrix is missing, build it from A, B, C, D scalars
        # Default to identity matrix values if scalars are missing
        val_A = A if A is not None else 1.0
        val_B = B if B is not None else 0.0
        val_C = C if C is not None else 0.0
        val_D = D if D is not None else 1.0
        
        self.matrix = np.array([[val_A, val_B], [val_C, val_D]])

    def get_matrix(self, A, B, C, D, thickness):
        """
        Returns the matrix constructed from the current optimization parameters.
        Note: 'thickness' is passed to satisfy the signature but affects only layout, not the matrix.
        """
        return np.array([[A, B], [C, D]])

    def get_sim_data(self):
        A, B = self.matrix[0]
        C, D = self.matrix[1]
        
        return (
            self.get_matrix, 
            lambda a, b, c, d, t: t,  # Length function just returns 't' (thickness)
            [float(A), float(B), float(C), float(D), float(self.thickness)]
        )