from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Any, ClassVar, get_type_hints

from .base import ElementBase, annotation_contains_node
from ..graph import Node, Parameter


_GRAPH_PARAMETER = object()


def parameter(default=None):
    """
    Explicitly declare an element field as a graph parameter.

    This is an alternative to annotating the field with Node.
    """
    return field(default=default, metadata={_GRAPH_PARAMETER: True})


@dataclass(eq=False, kw_only=True)
class OpticalElement(ElementBase):
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
    validate_graph_inputs: ClassVar[bool] = True

    # --------------------
    # CLASS INITIALIZATION
    # --------------------

    def __init_subclass__(cls, *, validate_graph_inputs: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.validate_graph_inputs = validate_graph_inputs

    def __post_init__(self):
        # Create a label if not explicitly set
        if self.label is None:
            cls = type(self)
            num_instances = OpticalElement._instance_counts.get(cls, 0)
            self.label = f"{cls.__name__}{num_instances + 1}"
            OpticalElement._instance_counts[cls] = num_instances + 1

    # -------------------
    # PARAMETER DISCOVERY
    # -------------------

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
            annotated = name in type_hints and annotation_contains_node(type_hints[name])
            if explicit or annotated:
                names.append(name)

        cls._parameter_names = tuple(names)
        return cls._parameter_names

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