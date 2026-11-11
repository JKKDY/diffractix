"""Unit constants for writing and reading SI values.

Diffractix computes in SI units (metres, radians). These constants are plain
scale factors: multiply to write a value, divide to read one back.

    from diffractix.units import mm, um, nm

    beam = GaussianBeam.from_waist(w0=500 * um, wavelength=1064 * nm)
    lens = ThinLens(f=50 * mm).variable(f=(20 * mm, 100 * mm))
    solution.value(lens.f) / mm                 # 47.3...
    units.format(solution.run().final.w, um)    # '45.02 um'

`50 * mm` is an ordinary float, so units add nothing to the graph or the
numerical backends. They do not check dimensions: `1 * mm + 1 * mrad` is allowed.
"""

import math

import numpy as np

__all__ = ["Unit", "m", "cm", "mm", "um", "nm", "pm", "rad", "mrad", "urad", "deg"]


class Unit(float):
    """A scale factor to SI that also knows its symbol, for formatting.

    Arithmetic with a Unit gives plain floats (or arrays), so `50 * mm` is 0.05.
    """

    def __new__(cls, scale: float, symbol: str):
        unit = super().__new__(cls, scale)
        unit.symbol = symbol
        return unit

    def __repr__(self) -> str:
        return f"Unit({float(self)!r}, {self.symbol!r})"


# Lengths
m = Unit(1.0, "m")
cm = Unit(1e-2, "cm")
mm = Unit(1e-3, "mm")
um = Unit(1e-6, "um")
nm = Unit(1e-9, "nm")
pm = Unit(1e-12, "pm")

# Angles
rad = Unit(1.0, "rad")
mrad = Unit(1e-3, "mrad")
urad = Unit(1e-6, "urad")
deg = Unit(math.pi / 180, "deg")


def format(value, unit: Unit, digits: int = 4) -> str:
    """Write an SI value in unit with `digits` significant digits, e.g. '45.02 um'.

    Arrays are written element by element: '[1.2, 3.4] mm'. Not in `__all__`, so
    `from ... import *` does not shadow the built-in format.
    """
    if not isinstance(unit, Unit):
        raise TypeError(f"unit must be a Unit such as mm, got {type(unit).__name__}.")
    if isinstance(digits, bool) or not isinstance(digits, int) or digits < 1:
        raise ValueError(f"digits must be a positive integer, got {digits!r}.")
    scaled = np.asarray(value, dtype=float) / unit
    if scaled.ndim == 0:
        return f"{float(scaled):.{digits}g} {unit.symbol}"
    numbers = ", ".join(f"{number:.{digits}g}" for number in scaled.ravel())
    return f"[{numbers}] {unit.symbol}"
