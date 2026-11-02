from ..graph import Symbol

from dataclasses import field

def make_symbol(key):
    """Create a dataclass field that initializes a Symbol for each instance.

    Accepts a symbol key or an existing Symbol.
    """
    if isinstance(key, Symbol):
        symbol_key = key.key
        has_default = key.has_default
        default = key.default
        return field(
            default_factory=lambda: (
                Symbol(symbol_key, default=default)
                if has_default
                else Symbol(symbol_key)
            )
        )
    else:
        return field(default_factory=lambda: Symbol(key))



AMBIENT_N = Symbol("ambient_n")
WAVELENGTH = Symbol("wavelength")
