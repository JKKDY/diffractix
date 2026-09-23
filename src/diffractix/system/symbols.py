from ..graph import Symbol

from dataclasses import field

def make_symbol(key):
    if isinstance(key, Symbol):
        return field(default_factory=lambda: Symbol(key.key))
    else:
        return field(default_factory=lambda: Symbol(key))



AMBIENT_N = Symbol("ambient_n")
WAVELENGTH = Symbol("wavelength")
