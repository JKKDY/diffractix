from ..graph import Symbol

from dataclasses import field

def make_symbol(key):
    """Create a dataclass field that initializes a Symbol for each instance.

    Accepts a symbol key or an existing Symbol.
    """
    if isinstance(key, Symbol):
        key = key.key
    return field(default_factory=lambda: Symbol(key))



AMBIENT_N = Symbol("ambient_n")
WAVELENGTH = Symbol("wavelength")
