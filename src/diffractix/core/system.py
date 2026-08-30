from __future__ import annotations

import inspect

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .beams.base import ParaxialState
from .core.system_vars import AMBIENT_N
from .elements.base import ElementBase
from .graph import Parameter


@dataclass(frozen=True)
class SourceInfo:
    """Location in user code where an element was added to the system."""

    file: str
    line: int
    call_index: int


@dataclass(frozen=True)
class Placement:
    """A single occurrence of an element in the optical path."""

    element: ElementBase
    z: float | None = None
    optimize_z: bool = False
    source_info: SourceInfo | None = None


class System:
    """
    Declarative description of an optical system.

    System contains the user-defined optical path, input state, system context,
    and requirements. build() resolves and compiles this description without
    mutating the System, its elements, or the parameter graph.
    """

    def __init__(self, ambient_n: float = 1.0, ambient_n_variable: bool = False):
        self.ambient_n = Parameter(value=ambient_n, name="ambient_n")
        if ambient_n_variable:
            self.ambient_n.variable()

        self.context = {
            AMBIENT_N.name: self.ambient_n,
        }

        self.beam: ParaxialState | None = None
        self._placements: list[Placement] = []
        self._requirements: list[Any] = []
        self._execution_counts: dict[tuple[str, int], int] = {}

    # --------
    # ELEMENTS
    # --------

    @property
    def placements(self) -> tuple[Placement, ...]:
        return tuple(self._placements)

    @property
    def elements(self) -> tuple[ElementBase, ...]:
        return tuple(placement.element for placement in self._placements)

    def add(self, element: ElementBase | Iterable[ElementBase], z: float | None = None, optimize_z: bool = False):
        """Add an element or sequence of elements to the optical path."""
        raise NotImplementedError

    def add_context(self, name: str, value: Node | Real):
        raise NotImplementedError


    # ----
    # BEAM
    # ----
    def add_input_beam(self, beam: ParaxialState):
        """Set the input beam or ray state for the system."""
        self.beam = beam
        return self


    # ------------
    # REQUIREMENTS
    # ------------
    @property
    def requirements(self) -> tuple[Any, ...]:
        return tuple(self._requirements)

    def require(self, *requirements):
        self._requirements.extend(requirements)
        return self


    # ----------
    # VALIDATION
    # ----------
    def _validate(self):
        """Validate the declarative system before resolution."""
        raise NotImplementedError


    # ----------
    # RESOLUTION
    # ----------
    def _resolve_elements(self):
        """Expand composites into concrete optical elements."""
        raise NotImplementedError

    def _resolve_layout(self, elements):
        """Resolve relative and absolute placement into a sequential optical path."""
        raise NotImplementedError

    def _resolve_refractive_indices(self, elements):
        """Resolve refractive-index inheritance and medium transitions."""
        raise NotImplementedError


    # -----------
    # COMPILATION
    # -----------
    def _compile(self, elements):
        """Compile resolved element expressions into the scalar parameter program."""
        raise NotImplementedError

    def _build_simulation(self, compiled):
        """Construct the executable Simulation from the compiled system."""
        raise NotImplementedError


    # -----
    # BUILD
    # -----
    def build(self):
        """Validate, resolve, compile, and return an independent Simulation."""
        self._validate()
        elements = self._resolve_elements()
        elements = self._resolve_layout(elements)
        elements = self._resolve_refractive_indices(elements)
        compiled = self._compile(elements)
        return self._build_simulation(compiled)

    # -----------
    # SOURCE INFO
    # -----------
    def _capture_source_info(self) -> SourceInfo:
        """Capture where an element was added in user code."""
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        key = (filename, lineno)
        call_index = self._execution_counts.get(key, 0)
        self._execution_counts[key] = call_index + 1

        return SourceInfo(
            file=filename,
            line=lineno,
            call_index=call_index,
        )