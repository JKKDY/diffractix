from abc import ABC, abstractmethod
from typing import Self, ClassVar


class ParaxialState(ABC):
    """State that can be propagated through a first-order ABCD optical system."""

    result_columns: ClassVar[tuple[str, ...]] = ()

    @property
    def execution_context(self):
        return {}

    @abstractmethod
    def propagate(self, A, B, C, D, n) -> Self:
        raise NotImplementedError
