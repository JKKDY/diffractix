from __future__ import annotations 
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class ElementInfo:
    """Inspection metadata for one resolved optical element."""

    type_name: str
    label: str | None
    path: str | None
    parameter_names: tuple[str, ...]
    parameter_indices: tuple[int, ...]


@dataclass(eq=False, frozen=True)
class SourceInfo:
    """Location in user code where an element was added to the system."""

    # currently in use for error messages
    file: str
    line: int
    call_index: int

    def __str__(self):
        return f"{self.file}:{self.line}"




@dataclass(frozen=True)
class ParameterInfo:
    """Descriptor for a simulation parameter."""

    parameter_id: int
    name: str | None
    value: float
    parameter_index: int | None
    lower_bound: float
    upper_bound: float
    owner_type: str | None = None
    owner_label: str | None = None

    @property
    def is_variable(self) -> bool:
        return self.parameter_index is not None



@dataclass(frozen=True)
class Placement:
    """A single occurrence of an element in the optical path."""

    element: ElementBase
    z: Node | Real | None = None
    source_info: SourceInfo | None = None
    path: str | None = None

    def describe(self, index: int | None = None) -> str:
        """Return a human-readable description of this placement."""
        element_type = type(self.element).__name__
        label = getattr(self.element, "label", None)

        if index is None:
            description = "Placement"
        else:
            description = f"Placement #{index}"

        if label:
            description += f" ({element_type} '{label}')"
        else:
            description += f" ({element_type})"

        if self.source_info is not None:
            description += (
                f" added at {self.source_info.file}:"
                f"{self.source_info.line}"
            )

        return description




@dataclass(frozen=True)
class SystemPlacement:
    """A sequential optical element with its resolved propagation medium."""

    placement: Placement
    refractive_index: Node

    @property
    def element(self) -> OpticalElement:
        return self.placement.element
