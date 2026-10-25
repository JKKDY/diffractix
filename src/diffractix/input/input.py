"""Declarative optical inputs and their numerical counterparts."""

from dataclasses import fields
from functools import cache

from ..core import OpticalBase
from ..graph import compile_ast


@cache
def _numerical_type(input_type):
    """Reuse one subclass whose constructors store numerical values directly."""
    if input_type.__dict__.get("_is_numerical", False):
        return input_type
    return type(
        f"Numerical{input_type.__name__}",
        (input_type,),
        {
            "__module__": input_type.__module__,
            "_is_numerical": True,
            "__setattr__": object.__setattr__,
            "_get_parameter_names": classmethod(lambda cls: ()),
        },
    )


class OpticalInput(OpticalBase):
    """Declarative optical input with graph-backed real scalar dataclass fields.

    Node annotations and parameter(...) declarations enable wrapping, exactly
    as for optical elements. All other fields stay numerical.

    Numerical subclasses inherit the dataclass constructor and __post_init__.
    Custom propagation should construct type(self) to retain the numerical type.
    """

    @classmethod
    def numerical_type(cls):
        """Return the cached class for direct numerical construction, including tracers.

        Supply numerical values explicitly for fields with symbolic defaults.
        """
        return _numerical_type(cls)

    def numeric(self, bindings=None):
        """Evaluate current graph fields in a new numerical dataclass instance.

        Explicit numerical Symbol bindings take precedence over Symbol defaults.
        Variable Parameters use their current values; no Simulation theta is used.
        Custom constructors must accept the dataclass init fields as keywords.
        Derived init=False fields are rebuilt by the numerical constructor.
        """
        input_fields = tuple(item for item in fields(self) if item.init)
        parameters = {
            item.name: getattr(self, item.name) for item in input_fields
            if item.name in self.parameter_names
        }
        roots = {
            name: handle for name, handle in parameters.items()
            if handle.node is not None
        }

        graph = compile_ast(tuple(roots.values()))
        
        resolved_bindings = {} if bindings is None else dict(bindings)
        for symbol in graph.symbols:
            if symbol.key not in resolved_bindings and symbol.has_default:
                resolved_bindings[symbol.key] = symbol.default
                
        values = dict.fromkeys(parameters)
        values.update(zip(
            roots,
            graph._evaluate_sequence(graph.initial_values,bindings=resolved_bindings)
        ))
        
        numerical = self.numerical_type()(**{
            item.name: values.get(item.name, getattr(self, item.name))
            for item in input_fields
        })

        return numerical
