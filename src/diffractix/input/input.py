"""Declarative optical inputs and their numerical counterparts."""

from dataclasses import dataclass, fields
from functools import cache

from diffractix.core import OpticalBase


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
class OpticalInput(OpticalBase):
    """Declarative optical input with graph-backed real scalar dataclass fields.

    Node annotations and parameter(...) declarations enable wrapping, exactly
    as for optical elements. All other fields stay numerical.

    Numerical subclasses inherit the dataclass constructor and __post_init__.
    Custom propagation should construct type(self) to retain the numerical type.
    Subclasses with additional dataclass fields must also use @dataclass.
    """

    label: str | None = None

    def __post_init__(self):
        self._init_label(self.label)

    @classmethod
    def numerical_type(cls):
        """Return the cached class for direct numerical construction, including tracers.

        Supply numerical values explicitly for fields with symbolic defaults.
        """
        return _numerical_type(cls)

    def numeric(self, bindings={}):
        """Evaluate current graph fields in a new numerical dataclass instance.

        Explicit numerical Symbol bindings take precedence over Symbol defaults.
        Variable Parameters use their current values; no Simulation theta is used.
        Custom constructors must accept the dataclass init fields as keywords.
        Derived init=False fields are rebuilt by the numerical constructor.
        """
        # get all fields (including purely numeric ones)
        constructor_fields = tuple(item for item in fields(self) if item.init)
        constructor_names = {field.name for field in constructor_fields}
        parameters = { name: getattr(self, name) for name in self.parameter_names if name in constructor_names}

        values = {}
        for item in constructor_fields:
            if item.name in parameters:
                handle = parameters[item.name]
                values[item.name] = None if handle.node is None else handle.evaluate(bindings)
            else:
                values[item.name] = getattr(self, item.name)

        return self.numerical_type()(**values)
