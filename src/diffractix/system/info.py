from __future__ import annotations 
from dataclasses import dataclass
from numbers import Real

from diffractix.core import OpticalBase
from diffractix.graph import Node

@dataclass(frozen=True)
class ConstraintData:
    position: int
    lower_bound: float
    upper_bound: float
    label: str | None
    description: str


@dataclass(eq=False, frozen=True)
class CallSite:
    """Location in user code where an element was added, used in error messages."""

    file: str
    line: int
    call_index: int

    def __str__(self):
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class ElementData:
    """Resolved element metadata; parameter indices address optics-program outputs."""

    element_id: int
    position: int
    type_name: str
    label: str | None
    path: str | None
    variable_parameter_ids: frozenset[int]
    parameter_names: tuple[str, ...] = ()
    parameter_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParameterData:
    parameter_id: int
    name: str | None
    value: float
    lower_bound: float
    upper_bound: float
    parameter_index: int | None



@dataclass(frozen=True)
class Placement:
    """An element added to a system, including its requested position and call site."""

    element: OpticalBase
    z: Node | Real | None = None
    call_site: CallSite | None = None
    path: str | None = None # identifies location within a composite element

    def describe(self, index: int | None = None) -> str:
        element_type = type(self.element).__name__
        label = getattr(self.element, "label", None)

        index_part = f" #{index}" if index is not None else ""
        label_part = f" '{label}'" if label else ""
        site_part = f" added at {self.call_site.file}:{self.call_site.line}" if self.call_site else ""

        return f"Placement{index_part} ({element_type}{label_part}){site_part}"


@dataclass(frozen=True)
class ResolvedPlacement:
    """A placement with its propagation medium resolved."""

    placement: Placement
    refractive_index: Node

    # TODO make sure  ResolvedPlacement.element is used and not ResolvedPlacement.placement.element
    @property
    def element(self) -> OpticalElement:
        return self.placement.element
