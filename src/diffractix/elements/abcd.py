"""
Defines the ABCD black-box element.
"""

from dataclasses import InitVar, dataclass

import numpy as np

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class ABCD(OpticalElement):
    """An element with a given ray-transfer matrix.

    Give the entries `A`, `B`, `C`, `D`, or the whole matrix as `matrix_val`.

    Parameters
    ----------
    A, B, C, D : float or Node, optional
        Matrix entries, acting on ``[x, theta]``: `B` in [m], `C` in [1/m].
        Default to the identity.
    thickness : float or Node, optional
        Element length [m]. Defaults to 0.
    n : float or Node, optional
        Refractive index after the element. Defaults to ``None``: the medium is
        unchanged.
    matrix_val : array_like, optional
        The whole matrix, shape ``(2, 2)``, instead of `A` to `D`.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"ABCD1"``.

    Raises
    ------
    ValueError
        If `matrix_val` is not 2 x 2, or `thickness` is a negative number.
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

        if self.n.node is not None:
            self.require(self.n > 0)

        length = self.element_length
        if isinstance(length, Node):
            self.require(length >= 0)
        elif length < 0:
            raise ValueError("Element length cannot be negative.")

    @property
    def matrix(self):
        """Ray-transfer matrix: ``((A, B), (C, D))``. Can also be set from a ``(2, 2)`` array."""
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
        """Element length [m]: `thickness`."""
        return self.thickness

    @property
    def refractive_index(self):
        """Refractive index: `n`, or ``None`` to keep the medium before the element."""
        return self.n
