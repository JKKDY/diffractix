from __future__ import annotations

import inspect
import autograd.numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

from diffractix.beams.base import ParaxialState
from diffractix.composites import CompositeElement
from diffractix.elements import OpticalElement, Interface, Space, ABCD
from diffractix.elements.base import ElementBase
from diffractix.graph import Node, Parameter, Literal, InputNode, compile_ast
from diffractix.simulation.simulation import Simulation, SimulationStep

from .errors import SystemValidationError
from .system_vars import AMBIENT_N

@dataclass(frozen=True)
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
    """Immutable descriptor for one simulation parameter."""

    index: int
    parameter: Parameter

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float

    owner_type: str | None = None
    owner_label: str | None = None




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


    def add(self, element: ElementBase | Iterable[ElementBase], z: Node | Real | None = None):
        """
        Add an element or sequence of elements to the optical path.

        Parameters
        ----------
        element:
            Element or sequence of elements to add. Composite elements are added
            as single declarative elements and expanded during System.build().
        z:
            Optional absolute position of the element along the optical path.
            May be a numerical scalar or graph expression. A variable Parameter
            makes the absolute position an independent optimization variable.
            If None, the element is placed directly after the preceding element.

        Returns
        -------
        System
            This system, allowing chained calls.
        """

        if isinstance(element, ElementBase):
            if z is not None and not isinstance(z, (Node, Real)):
                raise TypeError(
                    f"z must be a Node, numeric scalar, or None; got {type(z).__name__}."
                )

            if isinstance(z, bool):
                raise TypeError("z must be a Node, numeric scalar, or None.")

            self._placements.append(
                Placement(
                    element=element,
                    z=z,
                    source_info=self._capture_source_info(),
                )
            )
            return self

        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            if z is not None:
                raise ValueError(
                    "An absolute position cannot be applied to an element sequence. "
                    "Add the positioned element individually."
                )

            for child in element:
                self.add(child)

            return self

        raise TypeError(
            f"Expected an ElementBase or iterable of elements, got {type(element).__name__}."
        )


    def _capture_source_info(self) -> SourceInfo:
        """Capture where an element was added in user code."""
        frame = inspect.currentframe().f_back.f_back
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


    def add_context(self, name: str, value: Node | Real):
        """
        Add or replace a named value in the system build context.

        Parameters
        ----------
        name:
            Name used by SystemVar expressions to reference the context value.
        value:
            Numerical value or graph Node to associate with the name. Numerical
            values are converted to fixed system-owned Parameters.

        Returns
        -------
        System
            This system, allowing chained calls.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("Context name must be a non-empty string.")

        if not isinstance(value, (Node, Real)) or isinstance(value, bool):
            raise TypeError(
                f"Context value must be a Node or numeric scalar, got {type(value).__name__}."
            )

        if not isinstance(value, Node):
            value = Parameter(value=value, name=name)

        self.context[name] = value
        return self


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
        errors = []

        # BEAM
        if self.beam is None:
            errors.append("No input beam has been set. Use system.add_input_beam(beam) before build().")

        elif not isinstance(self.beam, ParaxialState):
            errors.append(f"Input beam must be a ParaxialState, got {type(self.beam).__name__}.")

        # validate context
        for name, value in self.context.items():
            if not isinstance(name, str) or not name:
                errors.append(f"Context key {name!r} must be a non-empty string.")

            if not isinstance(value, (Node, Real)) or isinstance(value, bool):
                errors.append(
                    f"Context value {name!r} must be a Node or numeric scalar, "
                    f"got {type(value).__name__}."
                )

        # validate refractive index
        try:
            ambient_n = self.ambient_n.value

            if not np.isfinite(ambient_n) or ambient_n <= 0:
                errors.append(f"Ambient refractive index must be finite and positive, got {ambient_n!r}.")

        except Exception as exc:
            errors.append(
                f"Ambient refractive index could not be evaluated: {exc}"
            )

        # validate placements (monotonically increasing placement)
        for i, placement in enumerate(self._placements):
            location_str = placement.describe(i)

            if not isinstance(placement.element, ElementBase):
                errors.append(
                    f"{location_str}: expected ElementBase, got "
                    f"{type(placement.element).__name__}."
                )
                continue

            if placement.z is not None:
                if not isinstance(placement.z, (Node, Real)) or isinstance(placement.z, bool):
                    errors.append(
                        f"{location_str}: absolute position z must be a Node, numeric scalar, "
                        f"or None; got {type(placement.z).__name__}."
                    )

                elif isinstance(placement.z, Real):
                    if not np.isfinite(placement.z):
                        errors.append(f"{location_str}: absolute position z must be finite, got {placement.z!r}.")

                    elif placement.z < 0:
                        errors.append(f"{location_str}: absolute position z cannot be negative, got {placement.z!r}.")

            element = placement.element

            if isinstance(element, CompositeElement):
                for path, leaf in element.walk():
                    try:
                        leaf._validate_for_build()

                    except Exception as exc:
                        errors.append(
                            f"{location_str}, child {path!r} "
                            f"({type(leaf).__name__} '{leaf.label}'): {exc}"
                        )

            elif isinstance(element, OpticalElement):
                try:
                    element._validate_for_build()

                except Exception as exc:
                    errors.append(f"{location_str}: {exc}")

        # REQUIREMENTS
        # TODO: validate requirement objects once the requirement API is defined.

        if errors:
            raise SystemValidationError(errors)


    # ----------
    # RESOLUTION
    # ----------
    def _resolve_elements(self):
        """Expand composite placements into concrete optical element placements."""
        resolved = []

        for placement in self._placements:
            element = placement.element

            if isinstance(element, OpticalElement):
                resolved.append(placement)
                continue

            if isinstance(element, CompositeElement):
                for i, (path, leaf) in enumerate(element.walk()):
                    resolved.append(
                        Placement(
                            element=leaf,
                            z=placement.z if i == 0 else None,
                            source_info=placement.source_info,
                            path=path,
                        )
                    )
                continue

            raise TypeError(
                f"{placement.describe()}: expected OpticalElement or CompositeElement, "
                f"got {type(element).__name__}."
            )

        return tuple(resolved)

    def _resolve_layout(self, placements):
        """
        Resolve relative and absolute placements into a sequential optical path.

        Absolute positions are converted into generated Space elements whose
        lengths depend on the accumulated upstream optical length. Relative
        placements are preserved directly.

        Parameters
        ----------
        placements:
            Sequence of flattened optical element placements.

        Returns
        -------
        tuple[Placement, ...]
            Sequential placements with all absolute positions resolved into
            generated spaces.
        """
        resolved = []
        current_z = Literal(0.0)

        for placement in placements:
            element = placement.element

            if placement.z is not None:
                target_z = placement.z

                if not isinstance(target_z, Node):
                    target_z = Literal(target_z)

                gap = target_z - current_z
                spacer = Space(d=gap,label=f"AutoSpace_to_{element.label}",)

                resolved.append(
                    Placement(
                        element=spacer,
                        source_info=placement.source_info,
                        path=placement.path,
                    )
                )

                current_z = current_z + spacer.element_length

            resolved.append(
                Placement(
                    element=element,
                    source_info=placement.source_info,
                    path=placement.path,
                )
            )

            current_z = current_z + element.element_length

        return tuple(resolved)


    def _resolve_refractive_indices(self, placements):
        """
        Resolve refractive-index inheritance and medium transitions.

        Elements without an explicit refractive index inherit the current medium.
        Interfaces update the current medium to their output refractive index.

        Parameters
        ----------
        placements:
            Sequential optical element placements after layout resolution.

        Returns
        -------
        tuple[SystemPlacement, ...]
            Sequential system placements with a resolved refractive index for
            every optical element.
        """
        resolved = []
        current_n = self.ambient_n

        for i, placement in enumerate(placements):
            element = placement.element
            index_handle = element.element_refractive_index

            if isinstance(index_handle, InputNode):
                index_node = index_handle.node
            else:
                index_node = index_handle

            if isinstance(element, Interface):
                try:
                    current_value = current_n.value
                    input_value = element.n1.value

                    if not np.isclose(current_value, input_value):
                        raise ValueError(
                            f"Refractive index mismatch at {placement.describe(i)}.\n"
                            f"Upstream medium: n={current_value:.4f}\n"
                            f"Interface input: n={input_value:.4f}\n"
                            "The Interface n1 must match the upstream refractive index."
                        )
                except AttributeError:
                    pass

                current_n = element.n2

            elif isinstance(element, ABCD) and index_node is not None:
                current_n = index_handle

            elif index_node is None:
                pass

            else:
                try:
                    current_value = current_n.value
                    target_value = index_handle.value

                    if not np.isclose(current_value, target_value):
                        raise ValueError(
                            f"Refractive index mismatch at {placement.describe(i)}.\n"
                            f"Upstream medium: n={current_value:.4f}\n"
                            f"Element requires: n={target_value:.4f}\n"
                            "Insert an Interface before this element."
                        )

                except AttributeError:
                    pass

                current_n = index_handle

            resolved.append(
                SystemPlacement(
                    placement=placement,
                    refractive_index=current_n,
                )
            )

        return tuple(resolved)


    # -----------
    # COMPILATION
    # -----------
    def _compile(self, elements):
        """
        Compile resolved element expressions into the scalar parameter program.

        Parameters
        ----------
        elements:
            Sequential system placements with resolved refractive indices.

        Returns
        -------
        tuple
            Compiled scalar graph, simulation steps, and location map.
        """
        roots = []
        steps = []
        location_map = {}

        def as_node(value):
            return value if isinstance(value, Node) else Literal(value)

        for system_placement in elements:
            element = system_placement.element
            matrix = element.matrix
            start = len(roots)
            step_index = len(steps)

            roots.extend((
                as_node(matrix[0][0]),
                as_node(matrix[0][1]),
                as_node(matrix[1][0]),
                as_node(matrix[1][1]),
                as_node(element.element_length),
                as_node(system_placement.refractive_index),
            ))

            steps.append(
                SimulationStep(
                    matrix_indices=(
                        (start, start + 1),
                        (start + 2, start + 3),
                    ),
                    length_index=start + 4,
                    refractive_index_index=start + 5,
                )
            )

            element_id = id(element)

            if element_id not in location_map:
                location_map[element_id] = []

            location_map[element_id].append(
                (step_index, step_index + 1)
            )

        graph = compile_ast(roots, context=self.context)

        parameter_info = []

        for index, parameter in enumerate(graph.variables):
            owner = parameter.owner

            parameter_info.append(
                ParameterInfo(
                    index=index,
                    parameter=parameter,
                    name=parameter.name,
                    initial_value=graph.initial_values[index],
                    lower_bound=parameter.lower_bound,
                    upper_bound=parameter.upper_bound,
                    owner_type=type(owner).__name__ if owner is not None else None,
                    owner_label=getattr(owner, "label", None) if owner is not None else None,
                )
            )

        location_map = {
            element_id: tuple(locations)
            for element_id, locations in location_map.items()
        }

        return graph, tuple(steps), tuple(parameter_info), location_map


    def _build_simulation(self, compiled):
        """
        Construct the executable Simulation from the compiled system.

        Parameters
        ----------
        compiled:
            Compiled scalar graph, simulation steps, and location metadata.

        Returns
        -------
        Simulation
            Executable simulation for the configured input beam.
        """
        graph, steps, parameter_info, location_map = compiled
        
        return Simulation(
            source=self.beam,
            graph=graph,
            steps=steps,
            parameter_info=parameter_info,
            location_map=location_map,
            requirements=self.requirements,
        )


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