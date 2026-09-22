from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class PlotPrimitive:
    label: str | None = None
    color: str | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class Line(PlotPrimitive):
    x: Any
    y: Any


@dataclass(frozen=True)
class ParametricLine(PlotPrimitive):
    evaluate: Callable[[Any], tuple[Any, Any]]
    t_min: float = 0.0
    t_max: float = 1.0
    samples: int = 100


@dataclass(frozen=True)
class VerticalLine(PlotPrimitive):
    x: float


@dataclass(frozen=True)
class Points(PlotPrimitive):
    x: Any
    y: Any


@dataclass(frozen=True)
class FillBetween(PlotPrimitive):
    x: Any
    lower: Any
    upper: Any


@dataclass(frozen=True)
class Box(PlotPrimitive):
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Polygon(PlotPrimitive):
    x: Any
    y: Any


@dataclass(frozen=True)
class ParametricShape(PlotPrimitive):
    evaluate: Callable[[Any], tuple[Any, Any]]
    t_min: float = 0.0
    t_max: float = 1.0
    samples: int = 100


@dataclass(frozen=True)
class Text(PlotPrimitive):
    x: float
    y: float
    text: str


Primitive = (
    Line
    | ParametricLine
    | VerticalLine
    | Points
    | FillBetween
    | Box
    | Polygon
    | ParametricShape
    | Text
)


