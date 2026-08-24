"""
Defines the ABCD black-box element.
"""

from dataclasses import InitVar, dataclass

import autograd.numpy as np

from .base import OpticalElement
from ..graph import Node


@dataclass(kw_only=True)
class ABCD(OpticalElement):
    """
    A black-box optical element defined by an arbitrary ABCD matrix.

    Parameters:
        A, B, C, D: Matrix coefficients.
        thickness: Physical length occupied by the element.
        n: Output refractive index. None means inherit the current medium.

    A complete matrix may alternatively be supplied through matrix_val.
    """

    A: Node = 1.0
    B: Node = 0.0
    C: Node = 0.0
    D: Node = 1.0

    thickness: Node = 0.0
    n: Node | None = None

    matrix_val: InitVar[np.ndarray | None] = None

    def __post_init__(self, matrix_val):
        super().__post_init__()

        if matrix_val is not None:
            self.matrix = matrix_val

    @property
    def matrix(self):
        return (
            (self.A, self.B),
            (self.C, self.D),
        )

    @matrix.setter
    def matrix(self, value):
        matrix = np.asarray(value)

        if matrix.shape != (2, 2):
            raise ValueError(
                "ABCD matrix must have shape (2, 2), "
                f"got {matrix.shape}."
            )

        self.A = matrix[0, 0]
        self.B = matrix[0, 1]
        self.C = matrix[1, 0]
        self.D = matrix[1, 1]

    @property
    def element_length(self):
        return self.thickness

    @property
    def element_refractive_index(self):
        return self.n