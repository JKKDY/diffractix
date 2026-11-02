"""Base state for paraxial optical inputs."""

from abc import ABC, abstractmethod
from typing import Self, ClassVar

from .input import OpticalInput


class ParaxialState(OpticalInput, ABC):
    """State that can be propagated through a first-order ABCD optical system."""

    result_columns: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        raise NotImplementedError
