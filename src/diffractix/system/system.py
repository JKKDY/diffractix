from __future__ import annotations

import copy
import inspect
import operator
import numpy as np
from collections.abc import Iterable, Hashable
from dataclasses import replace
from numbers import Real
from typing import Any
from types import MappingProxyType

from diffractix.input.paraxial_input import ParaxialState
from diffractix.composites import CompositeElement
from diffractix.elements import OpticalElement, Interface, Space, GaussianAperture
from diffractix.core import OpticalBase
from diffractix.graph import (
    Node, Parameter, Literal, InputNode, Symbol, compile_graph, precompile,
    walk_ast, evaluate_ast, Scalar,
)
from diffractix.simulation.simulation import Simulation
from diffractix.simulation.result import SimulationResultTracer
from diffractix.simulation.info import SimulationStep, SimulationInput

from .errors import SystemValidationError
from .symbols import AMBIENT_N
from .info import ElementData, ParameterData, Placement, CallSite, ResolvedPlacement


class System:
    """
    Declarative description of an optical system.

    System contains the user-defined optical path, optical input, system context,
    and design constraints. build() resolves and compiles this description into a simulation object
    """

    def __init__(
        self,
        *elements,
        optical_input: ParaxialState | None = None,
        ambient_n: Scalar | Parameter = 1.0,
    ):
        self._placements: list[Placement] = []
        self._constraints: list[Any] = []
        self._index_continuity_constraints: list[Any] = []
        self._execution_counts: dict[tuple[str, int], int] = {}
        self._bindings = {}

       
        self._ambient_n = None
        self.set_ambient_n(ambient_n)
        
        self._inputs = []

        if elements: self.add(*elements)
        if optical_input: self.add_input(optical_input)


    def _capture_call_site(self) -> CallSite:
        """Capture the user call to add() or the variadic constructor."""
        frame = inspect.currentframe().f_back.f_back
        if frame.f_code is System.__init__.__code__:
            frame = frame.f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        key = (filename, lineno)
        call_index = self._execution_counts.get(key, 0)
        self._execution_counts[key] = call_index + 1

        return CallSite(
            file=filename,
            line=lineno,
            call_index=call_index,
        )

    def set_ambient_n(self, ambient_n: Real | Parameter):
        """Set the ambient refractive index and update its symbol binding."""
        if not isinstance(ambient_n, Parameter):
            ambient_n = Parameter(value=ambient_n, name="ambient_n")

        value = ambient_n.value

        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Ambient refractive index must be finite and positive, got {value!r}.")

        self._ambient_n = ambient_n
        self.bind(AMBIENT_N.key, ambient_n)
        return self

    @property
    def ambient_n(self) -> Parameter:
        return self._ambient_n


    # ------------
    # ADD ELEMENTS
    # ------------
    @property
    def placements(self) -> tuple[Placement, ...]:
        return tuple(self._placements)

    @property
    def elements(self) -> tuple[OpticalBase, ...]:
        return tuple(placement.element for placement in self._placements)
    

    def add(
        self,
        *elements: OpticalBase | Iterable[OpticalBase],
        z: Node | Real | None = None,
        index: int | None = None,
    ):
        def flatten(items):
            for element in items:
                if isinstance(element, (OpticalElement, CompositeElement)):
                    yield element
                elif isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
                    yield from flatten(element)
                else:
                    raise TypeError(
                        "Expected an OpticalElement, CompositeElement, or iterable of elements, "
                        f"got {type(element).__name__}."
                    )

        elements = tuple(flatten(elements))

        if not elements:
            raise ValueError("At least one element is required.")
        if len(elements) > 1 and z is not None:
            raise ValueError("An absolute position can only be set for one element.")
        if index is not None:
            if isinstance(index, (bool, np.bool_)):
                raise TypeError("index must be an integer.")
            try:
                index = operator.index(index)
            except TypeError:
                raise TypeError("index must be an integer.") from None
            if index < 0:
                index = max(0, len(self._placements) + index)
            else:
                index = min(index, len(self._placements))
        if z is not None:
            if not isinstance(z, (Node, Real)) or isinstance(z, bool):
                raise TypeError(f"z must be a Node, Real, or None; got {type(z).__name__}.")

            if isinstance(z, Real):
                if not np.isfinite(z):
                    raise ValueError(f"Absolute position z must be finite, got {z!r}.")
                if z < 0:
                    raise ValueError(f"Absolute position z cannot be negative, got {z!r}.")

        for element in elements:
            dataclass_params = getattr(type(element), "__dataclass_params__", None)
            if dataclass_params is not None and dataclass_params.eq:
                raise TypeError(f"{type(element).__name__} must use @dataclass(eq=False).")

        call_site = self._capture_call_site()
        for element in elements:
            placement = Placement(element=element, z=z, call_site=call_site)

            if index is None:
                self._placements.append(placement)
            else:
                self._placements.insert(index, placement)
                index += 1

        return self

    def add_input(self, *inputs: ParaxialState | Iterable[ParaxialState]):
        """Set the optical input for the system."""

        if len(inputs) == 1 and not isinstance(inputs[0], ParaxialState) and isinstance(inputs[0], Iterable):
            inputs = tuple(inputs[0])

        if not inputs:
            raise ValueError("At least one input is required.")

        if len(inputs) + len(self._inputs) > 1:
            raise ValueError("Multiple optical inputs are not yet supported.")

        for input in inputs:
            if not isinstance(input, ParaxialState):
                raise TypeError(f"Expected a ParaxialState, got {type(input).__name__}.")
            if getattr(type(input), "_is_numerical", False):
                raise TypeError("System requires a declarative optical input, not a numerical input.")

        self._inputs.extend(inputs)
        return self

    # --------
    # BINDINGS
    # --------
    @property
    def bindings(self):
        return MappingProxyType(self._bindings)

    def bind(self, key: Hashable, value: Node | Scalar):
        try:
            hash(key)
        except TypeError:
            raise TypeError(
                f"Compile context key must be hashable, "
                f"got {type(key).__name__}."
            ) from None

        if not isinstance(value, (Node, Scalar)) or isinstance(value, bool):
            raise TypeError(
                "Compile context value must be a Node or numeric scalar, "
                f"got {type(value).__name__}."
            )

        if not isinstance(value, Node):
            if isinstance(value, Real):
                value = Parameter(value, name=key if isinstance(key, str) else None)
            elif isinstance(value, Scalar):
                value = Literal(value)

        self._bindings[key] = value
        return self



    # ------------
    # CONSTRAINTS
    # ------------
    @property
    def system_constraints(self) -> tuple:
        """Constraints added directly to the system."""
        return tuple(self._constraints)

    @property
    def constraints(self) -> tuple:
        return self._collect_constraints()

    def _collect_constraints(self, placements=()) -> tuple:
        """Collect declared and generated constraints once per optical object."""
        constraints = list(self._constraints)
        seen = set()

        def visit(element):
            if id(element) in seen:
                return
            seen.add(id(element))
            constraints.extend(element.constraints)
            if isinstance(element, CompositeElement):
                for child in element.elements:
                    visit(child)

        for optical_input in self._inputs or ():
            visit(optical_input)
        for element in self.elements:
            visit(element)
        for placement in placements:
            visit(placement.element)
        return tuple(constraints)

    def require(self, *constraints):
        self._constraints.extend(constraints)
        return self


    # ----------
    # VALIDATION
    # ----------
    def _validate(self):
        """Validate the declarative system before resolution."""
        errors = []

        if not self._inputs:
            errors.append("No input beam has been set. Use system.add_input(input) before build().")

        # validate placements (monotonically increasing placement)
        for i, placement in enumerate(self._placements):

            if isinstance(placement.element, CompositeElement):
                for path, leaf in placement.element.walk():
                    try:
                        leaf.validate()
                    except Exception as exc:
                        errors.append(
                            f"{placement.describe(i)}, child {path!r} "
                            f"({type(leaf).__name__} '{leaf.label}'): {exc}"
                        )
            elif isinstance(placement.element, OpticalElement):
                try:
                    placement.element.validate()
                except Exception as exc:
                    errors.append(f"{placement.describe(i)}: {exc}")

        if errors:
            raise SystemValidationError(errors)

    def _collect_bindings(self):
        """Collect bindings from the input, elements, and system."""
        bindings = {}
        seen = set()
        #? cant isntances of the same element have different bindings? maybe we should reject any form of duplciation directly 
        def add_binding(source):
            for key, value in source.items():
                if key in bindings:
                    raise ValueError(f"Duplicate symbol binding for {key!r}.")
                bindings[key] = value

        for optical_input in self._inputs:
            add_binding(optical_input.bindings)

        for element in self.elements:
            if id(element) in seen:
                continue
            seen.add(id(element))
            add_binding(element.bindings)

        add_binding(self._bindings)
        return bindings

    # ----------
    # RESOLUTION
    # ----------
    def _resolve_elements(self):
        """Expand composite placements into optical element placements."""
        resolved = []

        for placement in self._placements:

            if isinstance(placement.element, OpticalElement):
                resolved.append(placement)

            elif isinstance(placement.element, CompositeElement):
                for i, (path, leaf) in enumerate(placement.element.walk()):
                    resolved.append(Placement(
                        element=leaf,
                        z=placement.z if i == 0 else None,
                        call_site=placement.call_site,
                        path=path,
                    ))
            else:
                raise TypeError(
                    f"{placement.describe()}: expected OpticalElement or CompositeElement, "
                    f"got {type(placement.element).__name__}."
                )

        return tuple(resolved)

    def _resolve_layout(self, placements, bindings):
        resolved = []
        element_data = []
        current_z = 0.0

        def append(element, placement):
            resolved.append(Placement(
                element=element,
                call_site=placement.call_site,
                path=placement.path,
            ))

            parameters = tuple(
                (name, handle)
                for name, handle in zip(element.parameter_names, element.parameters)
                if handle.node is not None
            )

            parameter_ids = frozenset(
                id(node) for node in walk_ast((handle for _, handle in parameters), bindings)
                if isinstance(node, Parameter) and node.is_variable
            )

            element_data.append(ElementData(
                element_id=id(element),
                position=len(resolved) - 1,
                type_name=type(element).__name__,
                label=element.label,
                path=placement.path,
                variable_parameter_ids=parameter_ids,
                parameter_names=tuple(name for name, _ in parameters),
                element=element,
            ))

        for placement in placements:
            element = placement.element

            if placement.z is not None:
                gap = placement.z - current_z
                spacer = Space(d=gap, label=f"AutoSpace_to_{element.label}")

                append(spacer, placement)
                current_z = current_z + spacer.element_length

            append(element, placement)
            current_z = current_z + element.element_length

        return tuple(resolved), tuple(element_data),


    def _resolve_refractive_indices(self, placements, bindings):
        """Resolve the refractive index at each element and enforce continuity."""
        from diffractix.solver.constraint import Constraint

        resolved = []
        constraints = []
        current_n = self._inputs[0].refractive_index

        if isinstance(current_n, InputNode):
            current_n = current_n.node
        if current_n is None:
            current_n = self.ambient_n
        elif not isinstance(current_n, Node):
            current_n = Literal(current_n)

        for placement in placements:
            element = placement.element
            index = element.refractive_index

            if isinstance(index, InputNode):
                index = index.node

            # todo: document that Interface is special compared to other elements
            #? maybe we can find a better unified framework later? where elements can can define "specialities" 
            if isinstance(element, Interface):
                required_n = element.n1
                next_n = element.n2
            elif index is None:
                required_n = None
                next_n = current_n
            else:
                required_n = index
                next_n = index if isinstance(index, Node) else Literal(index)

            if required_n is not None:
                residual = current_n - required_n

                if residual.has_variable_parameters(bindings):
                    constraints.append(
                        Constraint(
                            residual,
                            lower_bound=0.0,
                            upper_bound=0.0,
                            label=f"Refractive index continuity at {placement.describe()}",
                        )
                    )
                elif not np.isclose(evaluate_ast(residual, bindings), 0.0):
                    raise ValueError(f"Refractive index mismatch at {placement.describe()}.")

            current_n = next_n
            resolved.append(
                ResolvedPlacement(
                    placement=placement,
                    refractive_index=current_n,
                )
            )

        return tuple(resolved), tuple(constraints)


    # -----------
    # COMPILATION
    # -----------
    def _compile_optics(self, placements, optical_inputs, bindings, continuity_constraints, element_data):
        """Compile optics and register variable dependencies of generated continuity.

        Theta follows first occurrence in optical roots, then additional
        continuity dependencies. ParameterData is returned in theta order.
        """
        roots = []
        steps = []
        inputs = []
        
          # compile elements
        for placement in placements:
            element = placement.element
            n = placement.refractive_index

            # TODO get rid of this differentiation
            matrix = (
                element._matrix_in_medium(n)
                if isinstance(element, GaussianAperture)
                else element.matrix
            )

            start = len(roots)
            roots.extend((
                matrix[0][0], matrix[0][1],
                matrix[1][0], matrix[1][1],
                element.element_length, n,
            ))

            steps.append(SimulationStep(
                matrix_indices=((start, start + 1), (start + 2, start + 3)),
                length_index=start + 4,
                refractive_index_index=start + 5,
            ))

        # compile inputs
        for optical_input in optical_inputs:
            field_indices = []

            for name, handle in zip(optical_input.parameter_names, optical_input.parameters):
                if handle.node is not None:
                    field_indices.append((name, len(roots)))
                    roots.append(handle)
                elif handle is optical_input.refractive_index:
                    field_indices.append((name, len(roots)))
                    roots.append(self.ambient_n)

            inputs.append(SimulationInput(
                initial_state=copy.deepcopy(optical_input.numeric(bindings)),
                field_indices=tuple(field_indices),
            ))

        compiled_element_data = []
        for placement, data in zip(placements, element_data, strict=True):
            start = len(roots)
            roots.extend(getattr(placement.element, name) for name in data.parameter_names)
            compiled_element_data.append(replace(
                data,
                parameter_indices=tuple(range(start, len(roots))),
            ))

        # some entries e.g. in matrix may be numbers -> convert to literals
        roots = [root if isinstance(root, Node) else Node._make_literal(root) for root in roots]
        parameters = {
            id(node): node for node in walk_ast(roots, bindings)
            if isinstance(node, Parameter) and node.is_variable
        }

        continuity_roots = (requirement.evaluate for requirement in continuity_constraints)
        for node in walk_ast(continuity_roots, bindings):
            if isinstance(node, Parameter) and node.is_variable and id(node) not in parameters:
                parameters[id(node)] = node
                roots.append(node)

        program = compile_graph(roots, bindings)
        if program.runtime_symbols:
            raise ValueError(f"Unresolved optical Symbol bindings: {tuple(program.runtime_symbols)!r}.")

        parameter_data = tuple(
            ParameterData(
                parameter_id=pid,
                name=parameter.name,
                value=parameter.value,
                lower_bound=parameter.lower_bound,
                upper_bound=parameter.upper_bound,
                parameter_index=program.parameter_indices[pid],
                parameter=parameter,
            )
            for pid, parameter in sorted(
                parameters.items(),
                key=lambda item: program.parameter_indices[item[0]],
            )
        )

        # theta = parameters in [roots] graph
        return program, steps, inputs, parameter_data, tuple(compiled_element_data)


    def _compile_constraints(self, constraints, bindings, parameter_indices, tracer=None):
        from diffractix.solver.constraint import prepare_constraints

        program, descriptors, callables = prepare_constraints(
            constraints, tracer=tracer, bindings=bindings,
        )
        if not program.parameter_indices.keys() <= parameter_indices.keys():
            raise ValueError("Constraints reference variable Parameters outside the optical model.")
        return program, descriptors, callables


    def build(self):
        """Resolve and compile the optical system into a simulation."""
        self._validate()
        bindings = self._collect_bindings()

        placements = self._resolve_elements()
        placements, element_data = self._resolve_layout(placements, bindings)
        placements, index_continuity = self._resolve_refractive_indices(placements, bindings)

        constraints = self._collect_constraints(placements)
       
        optics_program, stages, inputs, parameters_data, element_data = self._compile_optics(
            placements=placements,
            optical_inputs=self._inputs,
            bindings=bindings,
            continuity_constraints=index_continuity,
            element_data=element_data,
        )
        constraint_program, constraint_descriptors, callable_constraints = self._compile_constraints(
            constraints=(*constraints, *index_continuity),
            bindings=bindings,
            parameter_indices=optics_program.parameter_indices,
            tracer=SimulationResultTracer(element_data, inputs[0].initial_state),
        )

        return Simulation(
            element_data = element_data, 
            parameter_data=parameters_data,
            callable_constraints = callable_constraints, 
            constraint_descriptors = constraint_descriptors,
            constraint_program = constraint_program, 
            optics_program = optics_program,
            optical_stages = stages, 
            optical_inputs = inputs,
            bindings=bindings,
        )


    # ------
    # OUTPUT
    # ------
    def __str__(self) -> str:
        def format_node(val):
            if isinstance(val, InputNode):
                val = val.node
            if val is None:
                return "-"
            if isinstance(val, (Parameter, Literal)):
                return f"{val.value:.4g}"
            return f"{val:.4g}" if isinstance(val, Real) else str(val)

        def format_parameters(element):
            parts = []
            for name, handle in zip(element.parameter_names, element.parameters):
                node = handle.node
                if node is None:
                    parts.append(f"{name}=-")
                elif isinstance(node, Parameter):
                    state = "VAR" if node.is_variable else "FIX"
                    parts.append(f"{name}={node.value:.4g} [{state}]")
                else:
                    parts.append(f"{name}={node} [EXPR]")
            return ", ".join(parts) or "-"

        def format_constraint(constraint):
            if label := getattr(constraint, "label", None):
                return label
            if callable(constraint):
                return getattr(constraint, "__name__", None) or type(constraint).__name__
            return str(constraint)

        rows = [
            (
                str(i),
                "relative" if p.z is None else f"@{format_node(p.z)}",
                type(p.element).__name__,
                getattr(p.element, "label", None) or "-",
                format_parameters(p.element),
            )
            for i, p in enumerate(self.placements)
        ]

        headers = ("#", "Z", "Type", "Label", "Parameters")
        widths = [
            max(len(h), *(len(r[c]) for r in rows)) if rows else len(h)
            for c, h in enumerate(headers)
        ]

        col_gap = 4

        def format_row(row):
            return (" " * col_gap).join(value.ljust(width) for value, width in zip(row, widths))

        divider = "-" * (sum(widths) + col_gap * (len(widths) - 1))
        sources = ", ".join(type(source).__name__ for source in self._inputs) or "-"
        constraints = self.constraints

        lines = [
            "Optical System",
            "",
            format_row(headers),
            divider,
            *(format_row(r) for r in rows),
            divider,
            "",
            f"Source: {sources}",
            f"Ambient n: {self.ambient_n.value:.4g}",
            f"Constraints: {len(constraints)}",
        ]

        if constraints:
            lines.extend([
                "",
                "Constraints",
                "",
                *(f"{i}  {format_constraint(constraint)}" for i, constraint in enumerate(constraints)),
            ])

        return "\n".join(lines)
