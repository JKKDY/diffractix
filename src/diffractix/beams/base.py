from abc import ABC, abstractmethod
from typing import Self, ClassVar


class ParaxialState(ABC):
    """State that can be propagated through a first-order ABCD optical system."""

    result_type: ClassVar[type | None] = None

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        raise NotImplementedError