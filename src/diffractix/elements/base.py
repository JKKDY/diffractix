from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Any, ClassVar, get_args, get_type_hints

from ..graph import Node, Parameter, InputNode


_GRAPH_PARAMETER = object()


def parameter(default=None):
    """
    Explicitly declare an element field as a graph parameter.

    This is an alternative to annotating the field with Node.
    """
    return field(default=default, metadata={_GRAPH_PARAMETER: True})


def _annotation_contains_node(annotation) -> bool:
    """Return whether a type annotation contains Node or a Node subclass."""
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Node)
        except TypeError:
            return False

    return any(_annotation_contains_node(arg) for arg in get_args(annotation))


@dataclass(eq=False, kw_only=True)
class OpticalElement:
    """
    Base class for declarative optical elements.

    Graph-valued fields are declared either by annotating them with Node or
    explicitly using parameter(...). They are exposed through stable InputNode
    handles, allowing the graph behind an element parameter to be replaced
    without invalidating expressions that already reference the handle.
    """

    label: str | None = None

    _source_info: dict = field(default_factory=dict, init=False, repr=False)
    _requirements: list[Any] = field(default_factory=list, init=False, repr=False)

    # Static members
    _instance_counts: ClassVar[dict[type, int]] = {}
    _parameter_names: ClassVar[tuple[str, ...] | None] = None
    validate_graph_inputs: ClassVar[bool] = True


    # --------------------
    # CLASS INITIALIZATION
    # --------------------
    def __init_subclass__(cls, *, validate_graph_inputs: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.validate_graph_inputs = validate_graph_inputs
        cls._parameter_names = None

    def __post_init__(self):
        # Create a label if not explicitly set
        if self.label is None:
            cls = type(self)
            num_instances = OpticalElement._instance_counts.get(cls, 0)
            self.label = f"{cls.__name__}{num_instances + 1}"
            OpticalElement._instance_counts[cls] = num_instances + 1

        # Convert parameters to stable InputNode handles
        for name in self.parameter_names:
            value = self.__dict__.get(name)
            target = self._convert_to_node(name, value)
            object.__setattr__(self, name, InputNode(target))


    # -----------------------
    # PARAMETER DISCOVERY
    # -----------------------
    @classmethod
    def _get_parameter_names(cls) -> tuple[str, ...]:
        cached = cls.__dict__.get("_parameter_names")

        if cached is not None:
            return cached

        type_hints = get_type_hints(cls)
        names = []

        for dataclass_field in fields(cls):
            name = dataclass_field.name
            explicit = dataclass_field.metadata.get(_GRAPH_PARAMETER, False)
            annotated = name in type_hints and _annotation_contains_node(type_hints[name])

            if explicit or annotated:
                names.append(name)

        cls._parameter_names = tuple(names)
        return cls._parameter_names

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return type(self)._get_parameter_names()

    @property
    def parameters(self) -> tuple[InputNode, ...]:
        """Stable graph handles exposed by this element."""
        return tuple(getattr(self, name) for name in self.parameter_names)


    # --------------------
    # ATTRIBUTE ASSIGNMENT
    # --------------------
    def __setattr__(self, name, value):
        """
        Preserve graph-parameter handles when an element parameter is reassigned.

        During dataclass initialization raw values are assigned normally. Once a
        field has been wrapped in an InputNode, later assignments replace only
        the target of that handle.
        """
        current = self.__dict__.get(name)

        if isinstance(current, InputNode):
            current.node = self._convert_to_node(name, value)
            return

        object.__setattr__(self, name, value)

    def _convert_to_node(self, name: str, value: Node | Real | None) -> Node | None:
        """
        Convert an assigned value into an AST node.

        Existing Nodes retain their own identity and ownership. Raw numbers
        become element-local Parameters.
        """
        if isinstance(value, Node):
            return value

        if isinstance(value, Real) and not isinstance(value, bool):
            return Parameter(value=value, name=name, owner=self)

        if value is None:
            return None

        raise TypeError(
            f"Parameter {self._parameter_label(name)!r} must be a Node, "
            f"numeric scalar, or None; got {type(value).__name__}."
        )

    def _parameter_label(self, name: str) -> str:
        if self.label is None:
            return f"{type(self).__name__}.{name}"

        return f"{self.label}.{name}"


    # -----------------------
    # PARAMETER INTROSPECTION
    # -----------------------
    def _get_direct_parameter(self, name: str) -> Parameter | None:
        """Return the Parameter directly stored in a parameter slot."""
        handle = getattr(self, name)

        if not isinstance(handle, InputNode):
            raise TypeError(f"{self._parameter_label(name)} is not an InputNode.")

        if isinstance(handle.node, Parameter):
            return handle.node

        return None

    @property
    def values(self) -> tuple[float | None, ...]:
        """
        Current values of parameters when they can be evaluated directly.

        Unresolved graph parameters such as SystemVars or empty InputNodes are
        represented by None.
        """
        values = []

        for handle in self.parameters:
            try:
                values.append(handle.value)
            except (AttributeError, RuntimeError):
                values.append(None)

        return tuple(values)


    # -----------
    # VARIABILITY
    # -----------
    def _select_parameter_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if not names:
            return self.parameter_names

        unknown = [name for name in names if name not in self.parameter_names]

        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for {type(self).__name__}. "
                f"Available: {self.parameter_names}"
            )

        return names

    def variable(self, *names: str):
        """
        Mark directly stored Parameters as optimization variables.

        With no arguments, every parameter that directly contains a Parameter
        is marked variable. Named derived parameters raise rather than
        implicitly mutating Parameters elsewhere in the graph.
        """
        selected = self._select_parameter_names(names)
        strict = bool(names)

        for name in selected:
            parameter = self._get_direct_parameter(name)

            if parameter is not None:
                parameter.variable()
            elif strict:
                raise TypeError(
                    f"{self._parameter_label(name)} does not directly contain "
                    "a Parameter and cannot itself be marked variable."
                )

        return self

    def fixed(self, *names: str):
        """
        Mark directly stored Parameters as fixed.

        With no arguments, every parameter that directly contains a Parameter
        is marked fixed.
        """
        selected = self._select_parameter_names(names)
        strict = bool(names)

        for name in selected:
            parameter = self._get_direct_parameter(name)

            if parameter is not None:
                parameter.fixed()
            elif strict:
                raise TypeError(
                    f"{self._parameter_label(name)} does not directly contain "
                    "a Parameter and cannot itself be marked fixed."
                )

        return self


    # ------------
    # REQUIREMENTS
    # ------------
    @property
    def requirements(self) -> tuple[Any, ...]:
        """Persistent requirements attached to this element."""
        return tuple(self._requirements)

    def require(self, *requirements):
        self._requirements.extend(requirements)
        return self


    # ------------------
    # ELEMENT PROPERTIES
    # ------------------
    @property
    @abstractmethod
    def matrix(self):
        """
        Declarative 2x2 ABCD matrix.

        Entries may be graph Nodes or numerical constants. System.build()
        compiles these expressions into the numerical simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def element_length(self):
        """
        Declarative physical length of the element.

        May be a graph Node or a numerical constant.
        """
        raise NotImplementedError

    @property
    def element_refractive_index(self):
        """
        Declarative output refractive index.

        None means that the element leaves medium resolution to the surrounding
        system topology.
        """
        return None


    # ----------
    # VALIDATION
    # ----------
    def _validate_for_build(self):
        """
        Validate the element definition before compilation.

        Dependency tracing and undeclared numerical parameter detection will be
        implemented separately.
        """
        if not type(self).validate_graph_inputs:
            return

        # TODO: trace matrix, element_length, and element_refractive_index.
        # TODO: detect undeclared scalar numerical dependencies.
        return


    # -------
    # DISPLAY
    # -------
    def __str__(self):
        def current_value(value):
            if isinstance(value, Node):
                try:
                    return value.value
                except (AttributeError, RuntimeError):
                    return None

            if isinstance(value, Real):
                return value

            return None

        length = current_value(self.element_length)
        length_string = "?" if length is None else f"{length:.4g}"

        parameter_details = []

        for name, handle in zip(self.parameter_names, self.parameters):
            value = current_value(handle)
            value_string = "?" if value is None else f"{value:.4g}"
            node = handle.node

            if isinstance(node, Parameter):
                status = "VAR" if node.is_variable else "FIX"
            else:
                status = "EXPR"

            parameter_details.append(f"{name}={value_string} [{status}]")

        details = " | ".join(parameter_details)

        return (
            f"{type(self).__name__} '{self.label}' "
            f"L={length_string}"
            + (f" | {details}" if details else "")
        )
