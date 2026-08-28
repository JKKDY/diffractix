from abc import ABC, abstractmethod
from typing import Self


class ParaxialState(ABC):
    """State that can be propagated through a first-order ABCD optical system."""

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        raise NotImplementedError