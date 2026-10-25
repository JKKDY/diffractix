"""Base state for paraxial optical inputs."""

from abc import ABC, abstractmethod
from typing import Self, ClassVar

from .input import OpticalInput


class ParaxialState(OpticalInput, ABC):
    """State that can be propagated through a first-order ABCD optical system."""

    result_columns: ClassVar[tuple[str, ...]] = ()

    @property
    def execution_context(self):
        """Additional physical information provided by this state.

        Some simulations need information beyond the state variables
        being propagated. For example, a GaussianBeam provides its
        wavelength, which may be needed to evaluate the optical system.

        By default, no additional information is provided.
        """
        return {}

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        raise NotImplementedError
