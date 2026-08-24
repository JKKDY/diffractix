from __future__ import annotations

from abc import abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from numbers import Real
from typing import Any, ClassVar, get_args, get_type_hints

from ..graph import Node, Parameter, InputNode


_GRAPH_INPUT = "diffractix_graph_input"


def parameter(default=MISSING):
    """
    Explicitly declare an element field as a graph input.

    This is an alternative to annotating the field with Node.
    """
    if default is MISSING:
        return field(metadata={_GRAPH_INPUT: True})

    return field(
        default=default,
        metadata={_GRAPH_INPUT: True},
    )


def _annotation_contains_node(annotation) -> bool:
    """Return whether a type annotation contains Node or a Node subclass."""
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Node)
        except TypeError:
            return False

    return any(
        _annotation_contains_node(arg)
        for arg in get_args(annotation)
    )


@dataclass(eq=False, kw_only=True)
class OpticalElement:
    """
    Base class for declarative optical elements.

    Graph-valued fields are declared either by annotating them with Node or
    explicitly using parameter(...). They are exposed through stable InputNode
    handles, allowing the graph behind an element input to be replaced without
    invalidating expressions that already reference the handle.
    """

    label: str | None = None

    _source_info: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _requirements: list[Any] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    _label_counts: ClassVar[dict[type, int]] = {}
    _input_names_cache: ClassVar[tuple[str, ...] | None] = None
    validate_graph_inputs: ClassVar[bool] = True


    # -----------------------
    # CLASS INITIALIZATION
    # -----------------------
    def __init_subclass__(cls,*, validate_graph_inputs: bool = True,**kwargs):
        super().__init_subclass__(**kwargs)

        cls.validate_graph_inputs = validate_graph_inputs
        cls._input_names_cache = None

    def __post_init__(self):
        if self.label is None:
            cls = type(self)
            count = OpticalElement._label_counts.get(cls, 0) + 1
            OpticalElement._label_counts[cls] = count
            self.label = f"{cls.__name__}{count}"

        # Dataclass initialization has assigned the original field values by
        # this point -> convert declared graph inputs into AST nodes
        for name in self.input_names:
            value = self.__dict__.get(name)

            if isinstance(value, InputNode):
                continue

            target = self._make_input_target(name, value)
            object.__setattr__(self, name, InputNode(target))

        self._validate_graph_inputs()


    # ---------------------
    # GRAPH INPUT DISCOVERY
    # ---------------------
    @classmethod
    def _input_names(cls) -> tuple[str, ...]:
        if cls.__dict__.get("_input_names_cache") is not None:
            return cls._input_names_cache

        annotations = get_type_hints(cls)
        names = []

        for dataclass_field in fields(cls):
            name = dataclass_field.name

            explicit = dataclass_field.metadata.get(_GRAPH_INPUT, False)
            annotated = name in annotations and _annotation_contains_node(annotations[name])

            if explicit or annotated:
                names.append(name)

        cls._input_names_cache = tuple(names)
        return cls._input_names_cache

    @property
    def input_names(self) -> tuple[str, ...]:
        """Names of fields declared as graph inputs."""
        return type(self)._input_names()

    @property
    def inputs(self) -> tuple[InputNode, ...]:
        """Stable graph handles exposed by this element."""
        return tuple(
            getattr(self, name)
            for name in self.input_names
        )

    # -----------------------
    # ATTRIBUTE ASSIGNMENT
    # -----------------------
    def __setattr__(self, name, value):
        """
        Preserve graph-input handles when an element input is reassigned.

        During dataclass initialization the raw values are assigned normally.
        Once a field has been wrapped in an InputNode, later assignments replace
        only the target of that handle.
        """
        current = self.__dict__.get(name)

        if isinstance(current, InputNode):
            current.node = self._make_input_target(name, value)
            return

        object.__setattr__(self, name, value)

    def _make_input_target(self, name: str, value: Node | Real | None) -> Node | None:
        """
        Convert an assigned graph-input value into an AST target.

        Existing Nodes retain their own identity and ownership. Raw numbers
        become element-local Parameters.
        """
        if isinstance(value, Node):
            return value

        if isinstance(value, Real) and not isinstance(value, bool):
            return Parameter(
                value=float(value),
                name=name,
                owner=self,
            )

        if value is None:
            return None

        raise TypeError(
            f"Input {self._input_label(name)!r} must be a Node, "
            f"numeric scalar, or None; got {type(value).__name__}."
        )

    def _input_label(self, name: str) -> str:
        if self.label is None:
            return f"{type(self).__name__}.{name}"

        return f"{self.label}.{name}"

    # -----------------------
    # INPUT INTROSPECTION
    # -----------------------
    def _direct_parameter(self, name: str) -> Parameter | None:
        """
        Return the Parameter directly stored in an input slot.

        Expressions and references to other InputNodes are intentionally not
        followed: they are not parameters owned directly by this input.
        """
        handle = getattr(self, name)

        if not isinstance(handle, InputNode):
            raise TypeError(
                f"{self._input_label(name)} is not an InputNode."
            )

        if isinstance(handle.node, Parameter):
            return handle.node

        return None

    @property
    def variable_input_names(self) -> tuple[str, ...]:
        """Graph inputs that directly contain variable Parameters."""
        return tuple(
            name
            for name in self.input_names
            if (
                (parameter := self._direct_parameter(name)) is not None
                and parameter.is_variable
            )
        )

    @property
    def values(self) -> tuple[float | None, ...]:
        """
        Current values of graph inputs when they can be evaluated directly.

        Unresolved graph inputs such as SystemVars or empty InputNodes are
        represented by None.
        """
        values = []

        for handle in self.inputs:
            try:
                values.append(handle.value)
            except (AttributeError, RuntimeError):
                values.append(None)

        return tuple(values)

    # -----------------------
    # VARIABILITY
    # -----------------------
    def _select_input_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if not names:
            return self.input_names

        unknown = [
            name
            for name in names
            if name not in self.input_names
        ]

        if unknown:
            raise ValueError(
                f"Unknown graph input(s) {unknown} for "
                f"{type(self).__name__}. Available: {self.input_names}"
            )

        return names

    def variable(self, *names: str):
        """
        Mark directly stored Parameters as optimization variables.

        With no arguments, every input that directly contains a Parameter is
        marked variable. Named derived inputs raise rather than implicitly
        mutating Parameters elsewhere in the graph.
        """
        selected = self._select_input_names(names)
        strict = bool(names)

        for name in selected:
            parameter = self._direct_parameter(name)

            if parameter is not None:
                parameter.variable()
            elif strict:
                raise TypeError(
                    f"{self._input_label(name)} does not directly contain "
                    "a Parameter and cannot itself be marked variable."
                )

        return self

    def fixed(self, *names: str):
        """
        Mark directly stored Parameters as fixed.

        With no arguments, every input that directly contains a Parameter is
        marked fixed.
        """
        selected = self._select_input_names(names)
        strict = bool(names)

        for name in selected:
            parameter = self._direct_parameter(name)

            if parameter is not None:
                parameter.fixed()
            elif strict:
                raise TypeError(
                    f"{self._input_label(name)} does not directly contain "
                    "a Parameter and cannot itself be marked fixed."
                )

        return self

    # -----------------------
    # REQUIREMENTS
    # -----------------------
    @property
    def requirements(self) -> tuple[Any, ...]:
        """Persistent requirements attached to this element."""
        return tuple(self._requirements)

    def require(self, *requirements):
        self._requirements.extend(requirements)
        return self

    # -----------------------
    # DECLARATIVE OPTICS
    # -----------------------
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

    # -----------------------
    # VALIDATION
    # -----------------------
    def _validate_graph_inputs(self):
        """
        Validate undeclared Python state participating in optical expressions.

        Dependency tracing will be implemented separately. The hook lives here
        now so the element API does not need to change when validation is added.
        """
        if not type(self).validate_graph_inputs:
            return

        # TODO: trace matrix, element_length, and element_refractive_index.
        # TODO: detect undeclared scalar numerical dependencies.
        return

    # -----------------------
    # DISPLAY
    # -----------------------
    @staticmethod
    def _current_value(value):
        if value is None:
            return None

        if isinstance(value, Node):
            try:
                return value.value
            except (AttributeError, RuntimeError):
                return None

        if isinstance(value, Real):
            return value

        return None

    def __str__(self):
        length = self._current_value(self.element_length)
        length_string = "?" if length is None else f"{length:.4g}"

        parameter_details = []

        for name, value in zip(self.input_names, self.values):
            value_string = "?" if value is None else f"{value:.4g}"
            parameter = self._direct_parameter(name)

            if parameter is None:
                status = "EXPR"
            else:
                status = "VAR" if parameter.is_variable else "FIX"

            parameter_details.append(
                f"{name}={value_string} [{status}]"
            )

        details = " | ".join(parameter_details)

        return (
            f"{type(self).__name__} '{self.label}' "
            f"L={length_string}"
            + (f" | {details}" if details else "")
        )