from __future__ import annotations

from numbers import Real
from typing import ClassVar, get_args

from ..graph import InputNode, Node, Parameter


def annotation_contains_node(annotation) -> bool:
    """Return whether a type annotation contains Node or a Node subclass."""
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Node)
        except TypeError:
            return False
    return any(annotation_contains_node(arg) for arg in get_args(annotation))


class ElementBase:
    """Base class providing common graph-backed parameter behavior."""

    _parameter_names: ClassVar[tuple[str, ...] | None] = None


    # --------------------
    # CLASS INITIALIZATION
    # --------------------
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._parameter_names = None


    # -------------------
    # PARAMETER DISCOVERY
    # -------------------
    @classmethod
    def _get_parameter_names(cls) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return type(self)._get_parameter_names()

    @property
    def parameters(self) -> tuple[InputNode, ...]:
        return tuple(getattr(self, name) for name in self.parameter_names)

    @property
    def values(self) -> tuple[float | None, ...]:
        """Current values of parameters when they can be evaluated directly."""
        values = []
        for handle in self.parameters:
            try:
                values.append(handle.value)
            except (AttributeError, RuntimeError):
                values.append(None)
        return tuple(values)


    # --------------------
    # PARAMETER ASSIGNMENT
    # --------------------
    def __setattr__(self, name, value):
        if name in self.parameter_names:
            handle = self.__dict__.get(name)
            if isinstance(handle, InputNode):
                handle.node = self._convert_to_node(name, value)
                return
            value = InputNode(self._convert_to_node(name, value))
        object.__setattr__(self, name, value)

    def _convert_to_node(self, name: str, value: Node | Real | None) -> Node | None:
        if isinstance(value, Node):
            return value
        if isinstance(value, Real) and not isinstance(value, bool):
            return Parameter(
                value=value,
                name=name,
                owner=self,
            )
        if value is None:
            return None
        raise TypeError(
            f"Parameter {self._parameter_label(name)!r} must be a Node, numeric scalar, or None; "
            f"got {type(value).__name__}."
        )

    def _parameter_label(self, name: str) -> str:
        label = getattr(self, "label", None)
        if label is not None:
            return f"{label}.{name}"
        return f"{type(self).__name__}.{name}"

    def _get_direct_parameter(self, name: str) -> Parameter | None:
        handle = getattr(self, name)
        if not isinstance(handle, InputNode):
            raise TypeError(
                f"Parameter {self._parameter_label(name)!r} is not an InputNode."
            )
        if isinstance(handle.node, Parameter):
            return handle.node
        return None


    # -----------
    # VARIABILITY
    # -----------
    def _select_parameter_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if not names:
            return self.parameter_names
        unknown = [
            name
            for name in names
            if name not in self.parameter_names
        ]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for {type(self).__name__}. "
                f"Available: {self.parameter_names}"
            )
        return names

    def variable(self, *names: str):
        """
        Mark directly stored Parameters as optimization variables.

        With no names, every parameter directly containing a
        Parameter is marked variable. Derived parameters are skipped.
        """
        selected = self._select_parameter_names(names)
        strict = bool(names)
        for name in selected:
            parameter = self._get_direct_parameter(name)
            if parameter is not None:
                parameter.variable()
            elif strict:
                raise TypeError(
                    f"Parameter {self._parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked variable."
                )
        return self

    def fixed(self, *names: str):
        """
        Mark directly stored Parameters as fixed.

        With no names, every parameter directly containing a
        Parameter is marked fixed. Derived parameters are skipped.
        """
        selected = self._select_parameter_names(names)
        strict = bool(names)
        for name in selected:
            parameter = self._get_direct_parameter(name)
            if parameter is not None:
                parameter.fixed()
            elif strict:
                raise TypeError(
                    f"Parameter {self._parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked fixed."
                )
        return self
