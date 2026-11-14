"""
Defines the ABCD black-box element.
"""

from dataclasses import dataclass

import numpy as np

from .element import OpticalElement
from ..graph import Node


@dataclass(eq=False, kw_only=True)
class ABCD(OpticalElement):
    """An element with a given ray-transfer matrix.

    Give the entries `A`, `B`, `C`, `D`, or the whole matrix with :meth:`from_matrix`.

    Parameters
    ----------
    A, B, C, D : float or Node, optional
        Matrix entries, acting on ``[x, theta]``: `B` in [m], `C` in [1/m].
        Default to the identity.
    d : float or Node, optional
        Element length [m]. Defaults to 0.
    n : float or Node, optional
        Refractive index after the element. Defaults to ``None``: the medium is
        unchanged.
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"ABCD1"``.

    Raises
    ------
    ValueError
        If `d` is a negative number.

    Notes
    -----
    When `n` is given, requires ``n > 0``, which matters when it is variable.
    """

    A: Node = 1.0
    B: Node = 0.0
    C: Node = 0.0
    D: Node = 1.0

    d: Node = 0.0
    n: Node | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.n.node is not None:
            self.require(self.n > 0)

        self.require(self.element_length >= 0)

    @classmethod
    def from_matrix(cls, matrix, *, d=0.0, n=None, label=None) -> "ABCD":
        """Create an element from a whole ray-transfer matrix.

        Parameters
        ----------
        matrix : array_like
            The matrix ``((A, B), (C, D))``, shape ``(2, 2)``.
        d : float or Node, optional
            Element length [m]. Defaults to 0.
        n : float or Node, optional
            Refractive index after the element. Defaults to ``None``: the medium is
            unchanged.
        label : str, optional
            Name used in results and messages. Defaults to an automatic name.

        Returns
        -------
        ABCD
            The new element.

        Raises
        ------
        ValueError
            If `matrix` is not 2 x 2, or `d` is a negative number.

        Examples
        --------
        >>> from diffractix import ABCD
        >>> lens = ABCD.from_matrix([[1.0, 0.0], [-10.0, 1.0]])
        >>> float(lens.C.value)
        -10.0
        """
        matrix = np.asarray(matrix)
        if matrix.shape != (2, 2):
            raise ValueError(f"ABCD matrix must have shape (2, 2), got {matrix.shape}.")
        (A, B), (C, D) = matrix
        return cls(A=A, B=B, C=C, D=D, d=d, n=n, label=label)

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
        """Element length [m]: `d`."""
        return self.d

    @property
    def refractive_index(self):
        """Refractive index: `n`, or ``None`` to keep the medium before the element."""
        return self.n
