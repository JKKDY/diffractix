from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .scale import Scaling


@dataclass(frozen=True)
class OptimizationResult:
    """Optimization result; Solver returns x in physical SI coordinates."""

    x: np.ndarray
    success: bool
    cost: float
    message: str
    iterations: int | None = None
    raw: Any = None
    scaling: Scaling | None = None
