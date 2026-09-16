from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """Backend-independent numerical optimization result."""

    x: np.ndarray
    success: bool
    cost: float
    message: str
    iterations: int | None = None
    raw: Any = None
