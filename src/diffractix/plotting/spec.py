from __future__ import annotations

from dataclasses import dataclass

from .primitives import PlotPrimitive


@dataclass(frozen=True)
class PlotSpec:
    primitives: tuple[PlotPrimitive, ...]
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None