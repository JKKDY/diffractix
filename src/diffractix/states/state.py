"""Declarative optical inputs and their numerical counterparts."""

from dataclasses import dataclass, fields
from functools import cache

import numpy as np

from diffractix.core import OpticalBase
from diffractix.graph import Literal, Parameter, Symbol


@cache
def _numerical_type(input_type):
    """create subclass without parameter wrapping"""
    if input_type.__dict__.get("_is_numerical", False):
        return input_type
    return type(
        f"Numerical{input_type.__name__}",
        (input_type,),
        {
            "__module__": input_type.__module__,
            "_is_numerical": True,
            "__setattr__": object.__setattr__,
            "parameter_names": property(lambda self: ()),
        },
    )


@dataclass(eq=False, kw_only=True)
class OpticalState(OpticalBase):
    """Base class for the light at one position on the optical axis.

    Fields may be numbers or graph expressions (Parameters, Symbols), which are
    evaluated when the system is built and run. Results hold numerical copies,
    made by :meth:`numeric`.

    Notes
    -----
    To write a custom input, subclass `ParaxialState` with ``@dataclass``. Fields
    annotated with ``Node`` (or declared with ``parameter()``) accept numbers or
    graph expressions, as for elements; other fields stay plain values.
    """

    label: str | None = None

    def __post_init__(self):
        self._init_label(self.label)

    @classmethod
    def numerical_type(cls):
        """Return the numerical version of this class: the same state, built from numbers instead of graph expressions."""
        return _numerical_type(cls)

    def numeric(self, bindings={}):
        """Return a numerical copy of this state, with every field evaluated now.

        Parameters
        ----------
        bindings : Mapping, optional
            Values for Symbols, by key. Defaults to ``{}``.

        Returns
        -------
        OpticalState
            The same state holding numbers. Parameters use their current values,
            variable ones included.
        """
        # get all fields (including purely numeric ones)
        constructor_fields = tuple(item for item in fields(self) if item.init)
        constructor_names = {field.name for field in constructor_fields}
        parameters = { name: getattr(self, name) for name in self.parameter_names if name in constructor_names}

        values = {}
        for item in constructor_fields:
            if item.name in parameters:
                handle = parameters[item.name]
                values[item.name] = None if handle.node is None else handle.evaluate(bindings=bindings)
            else:
                values[item.name] = getattr(self, item.name)

        return self.numerical_type()(**values)

    def _for_display(self):
        """Return this state as numbers, or None if a field (such as n) has no value yet."""
        if getattr(type(self), "_is_numerical", False):
            return self
        if any(handle.node is None for handle in self.parameters):
            return None
        try:
            return self.numeric()
        except KeyError:
            return None

    def _describe_fields(self) -> str:
        """Describe the fields as written, for states that cannot be evaluated yet."""
        def number(value):
            if value is None:
                return "-"
            return f"{value:.4g}" if np.ndim(value) == 0 else f"array(shape={np.shape(value)})"

        def describe(node):
            if node is None:
                return "-"
            if isinstance(node, Symbol):
                return str(node.key)
            if isinstance(node, (Parameter, Literal)):
                return number(node.value)
            return repr(node)

        handles = dict(zip(self.parameter_names, self.parameters))
        parts = []
        for item in fields(self):
            if not item.init or item.name == "label":
                continue
            value = getattr(self, item.name)
            text = describe(handles[item.name].node) if item.name in handles else number(value)
            parts.append(f"{item.name}={text}")
        return f"{type(self).__name__}({', '.join(parts)})"
