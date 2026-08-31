from numbers import Real

import numpy as np

from ..beams.paraxial_state import ParaxialState
from ..composites.composite import CompositeElement
from ..elements import OpticalElement
from ..graph import Node



class SystemValidationError(ValueError):
    """Raised when a declarative System contains invalid input."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        message = "System validation failed:\n" + "\n".join(
            f"  {i}. {error}" for i, error in enumerate(errors, start=1)
        )
        super().__init__(message)