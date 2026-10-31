from __future__ import annotations

from dataclasses import fields, is_dataclass
from numbers import Real
from typing import ClassVar, get_args, get_type_hints

from diffractix.graph import InputNode, Node, Parameter


_GRAPH_PARAMETER = object()


def parameter(default=None):
    """Declare a dataclass field as a graph-backed real scalar parameter."""
    return field(default=default, metadata={_GRAPH_PARAMETER: True})


class OpticalBase:
    """Base class for optical objects with optional graph-backed parameters.

    Dataclass subclasses discover parameters from Node annotations or fields
    declared with parameter(). Non-dataclass subclasses can declare them as
    ``parameter_fields = ("name", ...)``.
    """

    _parameter_names: ClassVar[tuple[str, ...] | None] = None
    parameter_fields: ClassVar[tuple[str, ...] | None] = None
    
    def __new__(cls, *args, **kwargs):
        # use __new__ instead of __init__ so subclasses to have to call super().__init__ explicitely 
        instance = super().__new__(cls)
        instance._requirements = []
        return instance


    # -------------------
    # PARAMETER DISCOVERY
    # -------------------
    @staticmethod
    def _annotation_contains_node(annotation) -> bool:
        """Return whether an annotation contains Node or a Node subclass."""
        if isinstance(annotation, type):
            try:
                return issubclass(annotation, Node)
            except TypeError:
                return False
        return any(OpticalBase._annotation_contains_node(arg) for arg in get_args(annotation))


    @property
    def refractive_index(self):
        """Refractive index of an input or at an element's output.

        None leaves medium resolution to the surrounding system topology.
        Subclasses expose their own index without requiring a specific field name.
        """
        return None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of graph-backed fields declared explicitly or by dataclass metadata."""
        cls = type(self)
        cached = cls.__dict__.get("_parameter_names")

        if cached is not None:
            return cached

        explicit_names = getattr(cls, "parameter_fields", None)
        if explicit_names is not None:
            cls._parameter_names = tuple(explicit_names)
            return cls._parameter_names

        if is_dataclass(cls):
            type_hints = get_type_hints(cls)
            cls._parameter_names = tuple(
                item.name
                for item in fields(cls)
                if item.metadata.get(_GRAPH_PARAMETER, False)
                or cls._annotation_contains_node(type_hints.get(item.name))
            )
        else:
            cls._parameter_names = ()

        return cls._parameter_names

    @property
    def parameters(self) -> tuple[InputNode, ...]:
        return tuple(getattr(self, name) for name in self.parameter_names)

    @property
    def values(self) -> tuple[float | None, ...]:
        """Return current parameter values, or None if unavailable."""
        values = []
        for handle in self.parameters:
            try:
                values.append(handle.value)
            except (AttributeError, RuntimeError):
                values.append(None)
        return tuple(values)


    # ------------
    # REQUIREMENTS
    # ------------
    @property
    def requirements(self) -> tuple:
        """Requirements attached to this optical object."""
        return tuple(self._requirements)

    def require(self, *requirements):
        """Attach requirements to this optical object."""
        self._requirements.extend(requirements)
        return self


    # --------------------
    # PARAMETER ASSIGNMENT
    # --------------------
    def __setattr__(self, name, value):
        def convert_value_to_node(name: str) -> Node | None:
            if isinstance(value, Node):
                return value
            if isinstance(value, Real) and not isinstance(value, bool):
                return Parameter(value=value, name=name, owner=self)
            if value is None:
                return None
            raise TypeError(
                f"Parameter {self._get_parameter_label(name)!r} must be a Node, numeric scalar, or None; "
                f"got {type(value).__name__}."
            )

        if name in self.parameter_names:
            handle = self.__dict__.get(name)
            if isinstance(handle, InputNode):
                handle.node = convert_value_to_node(name)
                return
            value = InputNode(convert_value_to_node(name,))
        object.__setattr__(self, name, value)

    def _get_parameter_label(self, name: str) -> str:
        label = getattr(self, "label", None)
        if label is not None:
            return f"{label}.{name}"
        return f"{type(self).__name__}.{name}"

    def _get_parameter_node(self, name: str) -> Parameter | None:
        handle = getattr(self, name)
        if not isinstance(handle, InputNode):
            raise TypeError( f"Parameter {self._get_parameter_label(name)!r} is not an InputNode.")
        if isinstance(handle.node, Parameter):
            return handle.node
        return None


    # -----------
    # VARIABILITY
    # -----------
    def _validate_parameter_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if not names:
            return self.parameter_names

        unknown = [name for name in names if name not in self.parameter_names]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for {type(self).__name__}. "
                f"Available: {self.parameter_names}"
            )

        return names

    def variable(self, *parameter_names: str):
        """
        Mark directly stored Parameters as optimization variables.

        With no names, every parameter directly containing a
        Parameter is marked variable. Derived parameters are skipped.
        """
        strict = bool(parameter_names)
        names = self._validate_parameter_names(parameter_names)

        for name in names:
            parameter = self._get_parameter_node(name)
            if parameter is not None:
                parameter.variable()
            elif strict:
                raise TypeError(
                    f"Parameter {self._get_parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked variable."
                )

        return self

    def fixed(self, *names: str):
        """
        Mark directly stored Parameters as fixed.

        With no names, every parameter directly containing a
        Parameter is marked fixed. Derived parameters are skipped.
        """        
        strict = bool(parameter_names)
        names = self._validate_parameter_names(parameter_names)

        for name in names:
            parameter = self._get_parameter_node(name)
            if parameter is not None:
                parameter.fixed()
            elif strict:
                raise TypeError(
                    f"Parameter {self._get_parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked fixed."
                )

        return self
