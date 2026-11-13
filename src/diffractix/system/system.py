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

from diffractix.states.paraxial_state import ParaxialState
from diffractix.composites import CompositeElement
from diffractix.elements import OpticalElement, Interface, Space, GaussianAperture
from diffractix.core import OpticalBase
from diffractix.graph import (
    Node, Parameter, Literal, InputNode, compile_graph, walk_ast, evaluate_ast, clone_ast, Scalar,
)
from diffractix.simulation.simulation import Simulation
from diffractix.simulation.result import SimulationResultTracer
from diffractix.simulation.info import SimulationStep, SimulationInput

from .errors import SystemValidationError
from .symbols import AMBIENT_N
from .info import ElementData, ParameterData, Placement, CallSite, ResolvedPlacement



class System:
    """A sequence of elements that light propagates through, together with design constraints.

    A system specifies the optical design; :meth:`build` turns it into a `Simulation`
    that can be run and optimized. Elements without a position follow the
    previous element directly. Elements given a position with ``z`` start there.
    The gap before them is filled with a `Space` with refractive index `ambient_n`.

    Parameters
    ----------
    *elements : OpticalElement, CompositeElement or iterable of them
        Elements to add, in propagation order. See :meth:`add`.
    optical_input : Ray, RayBundle or GaussianBeam, optional
        Light entering the system. See :meth:`add_input`. Defaults to ``None``; one
        is required before :meth:`build`.
    ambient_n : float or Parameter, optional
        Ambient refractive index: the medium around the elements. Defaults to 1.0.

    Examples
    --------
    >>> from diffractix import Ray, Space, System, ThinLens
    >>> from diffractix.units import mm
    >>> system = System(
    ...     ThinLens(f=100 * mm), Space(d=100 * mm),
    ...     optical_input=Ray(x=1 * mm, theta=0),
    ... )
    >>> round(float(system.build().run().final.x), 9)
    0.0
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
        """Set the ambient refractive index.

        Parameters
        ----------
        ambient_n : float or Parameter
            Refractive index of the medium around the elements. A number becomes a
            fixed `Parameter` named ``"ambient_n"``.

        Returns
        -------
        System
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If the index is not finite and positive.
        """
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
        """Ambient refractive index: the medium around the elements.

        Composites use it as their default `ambient_n`, and light entering the
        system without its own ``n`` starts in it.
        """
        return self._ambient_n


    # ------------
    # ADD ELEMENTS
    # ------------
    @property
    def placements(self) -> tuple[Placement, ...]:
        """Placements: one entry per added element or composite, with its requested position, in propagation order."""
        return tuple(self._placements)

    @property
    def elements(self) -> tuple[OpticalBase, ...]:
        """Elements: the added elements and composites, in propagation order.

        Composites are not expanded into their constituent parts.
        """
        return tuple(placement.element for placement in self._placements)
    

    def add(
        self,
        *elements: OpticalBase | Iterable[OpticalBase],
        z: Node | Real | None = None,
        at: str | OpticalBase | Node | Real | None = None,
        index: int | None = None,
    ):
        """Add elements, optionally at an absolute position.

        Parameters
        ----------
        *elements : OpticalElement, CompositeElement or iterable of them
            Elements to add, in propagation order. A list or tuple is unpacked
            into its elements, so passing one is the same as passing its
            contents directly.
        z : float or Node, optional
            Position along the axis [m] of one point of the element, chosen by `at`.
            Requires a single element. Defaults to ``None``: the element follows the
            previous one directly.
        at : str, OpticalElement, CompositeElement, float or Node, optional
            Which point of the element `z` refers to: ``"front"``, ``"center"`` or
            ``"back"``; a part of the added composite, meaning that part's front; or a
            length [m] measured from the element's front. Requires `z`. Defaults to
            ``"front"``.
        index : int, optional
            Position in the list of placements to insert at; negative values count
            from the end. Defaults to appending.

        Returns
        -------
        System
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If an item is not an element or composite, or if `z`, `at` or `index` has
            the wrong type.
        ValueError
            If no element is given, `z` is given with several elements, `at` is given
            without `z`, `at` is not part of the added composite, or the resulting
            front position is negative.

        Notes
        -----
        A position given with `z` must not lie before the end of the preceding
        element: :meth:`build` raises an error if the gap before it would be
        negative.

        Examples
        --------
        A lens placed at an absolute position gets a generated gap in front of it:

        >>> from diffractix import Ray, System, ThinLens
        >>> from diffractix.units import mm
        >>> lens = ThinLens(f=50 * mm)
        >>> system = System(optical_input=Ray(x=0, theta=0)).add(lens, z=30 * mm)
        >>> round(float(system.build().run()[lens].z_start) / mm, 9)
        30.0
        """
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
        if at is not None and z is None:
            raise ValueError("at requires an absolute position z.")
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

            if isinstance(z, Real) and not np.isfinite(z):
                raise ValueError(f"Absolute position z must be finite, got {z!r}.")

            # Normalize: from here on z is where the element's front goes.
            if isinstance(at, str):
                points = {"front": lambda length: 0, "center": lambda length: length / 2, "back": lambda length: length}
                if at not in points:
                    raise ValueError(f"at must be one of {tuple(points)}, got {at!r}.")
                at = points[at](elements[0].element_length)
            elif isinstance(at, (OpticalElement, CompositeElement)):
                # Length in front of `at` inside the added composite; for a nested composite
                # that is the front of its first leaf.
                leaves = elements[0].leaf_elements if isinstance(elements[0], CompositeElement) else ()
                first = at.leaf_elements[:1] if isinstance(at, CompositeElement) else (at,)
                matches = [i for i, leaf in enumerate(leaves) if first and leaf is first[0]]
                if at is elements[0] or not matches:
                    raise ValueError(f"{at!r} is not part of {elements[0]!r}.")
                if len(matches) > 1:
                    raise ValueError(
                        f"{at!r} appears more than once in {elements[0]!r}; its position is ambiguous."
                    )
                at = sum(leaf.element_length for leaf in leaves[:matches[0]])
            if at is not None:
                if isinstance(at, bool) or not isinstance(at, (Node, Real)):
                    raise TypeError(
                        f'at must be "front", "center", "back", an element or a length; got {type(at).__name__}.'
                    )
                if isinstance(at, Real) and not np.isfinite(at):
                    raise ValueError(f"at must be finite, got {at!r}.")
                if not (isinstance(at, Real) and at == 0):
                    z = z - at
            if isinstance(z, Real):
                if z < 0:
                    raise ValueError(
                        f"Absolute position z cannot be negative, got {z!r}"
                        + ("" if at is None else " for the element's front (after applying at)")
                        + "."
                    )

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
        """Set the light entering the system.

        Parameters
        ----------
        *inputs : Ray, RayBundle or GaussianBeam, or an iterable of one
            The light state. Only one input is currently supported.

        Returns
        -------
        System
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If an input is not a `Ray`, `RayBundle` or `GaussianBeam`, or is a
            numerical state rather than one built by the user.
        ValueError
            If no input is given, or the system would have more than one.
        """

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
        """Bindings: values the system provides for Symbols, by key.

        This includes ``"ambient_n"``. Elements and the input add their own bindings
        at :meth:`build`.
        """
        return MappingProxyType(self._bindings)

    def bind(self, key: Hashable, value: Node | Scalar):
        """Provide a value for a Symbol.

        Elements can depend on values that are only known in the system, such as the
        ambient index or the wavelength; they refer to them with a `Symbol`, which is
        replaced by the bound value at :meth:`build`.

        Parameters
        ----------
        key : hashable
            Key of the Symbol, for example ``"wavelength"``.
        value : float, complex or Node
            The value. A real number becomes a fixed `Parameter` named after a string
            key; a complex number becomes a constant.

        Returns
        -------
        System
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If `key` is not hashable or `value` is not a number or Node.
        """
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
        """Constraints: those added to the system with :meth:`require`."""
        return tuple(self._constraints)

    @property
    def constraints(self) -> tuple:
        """Constraints: those of the system, its input, and its elements and their parts.

        Constraints that :meth:`build` generates, such as refractive index continuity,
        are not included.
        """
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

    def require(self, *constraints, label: str | None = None):
        """Attach constraints that must hold in any optimization of this system.

        Parameters
        ----------
        *constraints : Comparison, Constraint or callable
            Comparisons such as ``space.d <= 100 * mm``, `Constraint` objects, or
            callables without arguments that return a comparison.
        label : str, optional
            Name for a single comparison, used in reports.

        Returns
        -------
        System
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If `label` is given with several constraints or with a `Constraint`.

        Notes
        -----
        Callables must be differentiable by the numerical backend; diffractix does
        not check this.
        """
        from diffractix.solver.constraint import label_constraints
        self._constraints.extend(label_constraints(constraints, label))
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
        def add_binding(source):
            for key, value in source.items():
                if key in bindings:
                    raise ValueError(f"Duplicate symbol binding for {key!r}.")
                bindings[key] = value

        for optical_input in self._inputs:
            add_binding(optical_input.bindings)

        for element in self.elements:
            leaves = element.leaf_elements if isinstance(element, CompositeElement) else (element,)
            for leaf in leaves:
                if id(leaf) not in seen:
                    seen.add(id(leaf))
                    add_binding(leaf.bindings)

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
                        path=(placement.element.label, *path.split(".")),
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

        def append(element, placement, path):
            resolved.append(Placement(
                element=element,
                call_site=placement.call_site,
                path=path,
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
                path=path,
                variable_parameter_ids=parameter_ids,
                parameter_names=tuple(name for name, _ in parameters),
                element=element,
            ))

        for placement in placements:
            element = placement.element

            if placement.z is not None:
                gap = placement.z - current_z
                spacer = Space(d=gap, label=f"AutoSpace_to_{element.label}")

                append(spacer, placement, path=None)  # a generated gap belongs to no composite
                current_z = current_z + spacer.element_length

            append(element, placement, placement.path)
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
                    incoming = float(evaluate_ast(current_n, bindings))
                    required = float(evaluate_ast(required_n, bindings))
                    origin = "the optical input" if not resolved else "the preceding element"
                    hint = (
                        " Give the optical input a matching n, or change the system's ambient_n."
                        if not resolved else ""
                    )
                    raise ValueError(
                        f"Refractive index mismatch at {placement.describe()}: the light arrives "
                        f"in n = {incoming:.6g} (from {origin}) but "
                        f"{type(element).__name__} '{element.label}' expects n = {required:.6g}.{hint}"
                    )

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
        """Validate the system and compile it into a simulation.

        Returns
        -------
        Simulation
            The compiled system. It is a snapshot: later changes to the system, its
            elements or their parameters do not affect it. Call `build` again to
            reflect those changes.

        Raises
        ------
        SystemValidationError
            If the system has no input or an element is invalid. All problems are
            listed together.
        ValueError
            If the refractive index the light arrives in does not match what an
            element expects, two sources bind the same Symbol, a Symbol has no value,
            or a constraint is constant and violated.
        """
        self._validate()
        # Compile from a structural copy of the bound expressions and keep that copy in
        # the Simulation, so later edits to the live graph reach neither.
        memo = {}
        bindings = {
            key: clone_ast(value, memo) if isinstance(value, Node) else value
            for key, value in self._collect_bindings().items()
        }

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
