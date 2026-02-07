"""
Defines the ABCD 'Black Box' element.
"""
from dataclasses import dataclass, field, InitVar
import autograd.numpy as np
from .element import OpticalElement
from ..graph import Node, Parameter, Symbol


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
        n (float): Refractive index. Optional. if not set will inherit from previous element. If that is not set, defaults to 1.0.
        A, B, C, D (float): Individual matrix components (optional helpers).

    """
    A: Node = 1.0
    B: Node = 0.0
    C: Node = 0.0
    D: Node = 1.0
    thickness: Node = 0.0 
    n : Node = None 

    matrix_val: InitVar[np.ndarray] = None


    def __post_init__(self, matrix_val):
        if matrix_val is not None:
            self.matrix = np.array(matrix_val)
        super().__post_init__()

    @property
    def matrix(self):
        """Getter: Reconstructs matrix from current params."""
        return np.array([[self.A.value, self.B.value], [self.C.value, self.D.value]])

    @matrix.setter
    def matrix(self, mat: np.ndarray):
        assert mat.shape == (2,2), f"Mismatched matrix shape. Expected shape=(2,2), got {mat.shape}"
        self.A = mat[0,0]
        self.B = mat[0,1]
        self.C = mat[1,0]
        self.D = mat[1,1]

    @property
    def element_length(self):
        return self.thickness

    @property
    def element_refractive_index(self):
        return self.n

    def init_placeholders(self, environment: "Environment"):
        if isinstance(self.n, PlaceHolder):
            self.n.value = environment.ambient_n.value
            self.n.fixed = environment.ambient_n.fixed

    def compute_matrix(self, A, B, C, D):
        return np.array([[A, B], [C, D]])

