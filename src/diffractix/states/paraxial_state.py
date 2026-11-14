"""Base state for paraxial optical inputs."""

from abc import ABC, abstractmethod
from typing import Self, ClassVar

from .state import OpticalState


class ParaxialState(OpticalState, ABC):
    """Base class for light described by ray-transfer (ABCD) matrices.

    `Ray`, `RayBundle` and `GaussianBeam` are the built-in kinds. Subclass it to
    add another.

    Attributes
    ----------
    result_columns : tuple of str
        Quantities shown for each stage when a result is printed.
    """

    result_columns: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        """Return the state after a first-order (ABCD) transformation.

        Parameters
        ----------
        A, B, C, D : float
            Elements of the ray-transfer matrix.
        n : float
            Refractive index after the transformation.

        Returns
        -------
        ParaxialState
            The new state, of the same type; ``self`` is unchanged.
        """
        raise NotImplementedError
