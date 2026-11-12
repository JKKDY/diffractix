# Docstring style

The public API (what `diffractix.__all__` exports, and the objects it returns) has
NumPy-style docstrings. They are API documentation, not a textbook: say what a
thing is and how to use it, briefly. Derivations belong in the theory pages.
`src/diffractix/input/` is the reference implementation of this guide.

## Wording

- Plain, direct English. Short sentences, common words, no filler ("note that",
  "simply", "it should be noted").
- Describe behaviour, not implementation. Internal mechanisms (graph wrapping,
  tracer handling, caching) only where a user is affected, and then in user terms.
- Say what is exact: signs, defaults, edge values (`inf`, `0`, NaN), and which
  index or wavelength is meant.
- Avoid vague references such as "this plane" or "the current position" unless
  the class says what they are. A light state describes one position on the
  optical axis; use "the beam's position" or "where the beam starts".

## Units

- SI everywhere. Put the unit in brackets right after the quantity name:
  `Waist radius [m].`, `Distance [m] along the axis ...`, `Angle to the optical axis [rad]: ...`.
- Dimensionless quantities get no brackets.

## Summary lines

- Classes: a noun phrase. `A single paraxial ray: its height and angle at one position on the optical axis.`
- Methods and functions: a verb. `Return ...`, `Create ...`, `Plot ...`.
- Properties: `Name [unit]: what it is.` For example
  `Rayleigh range [m]: the distance from the waist over which the area doubles.`
  Add one short clause for signs or edge cases if needed.

## Sections

Use only the sections that add something, in this order: Parameters, Returns,
Raises, See Also, Notes, Examples.

- **Parameters**: `name : type` then the description. Types: `float`, `int`,
  `str`, `bool`, `array_like`, `float or Node` where a graph expression is
  accepted, `optional` for parameters with defaults. Every optional parameter
  ends with `Defaults to ...`.
- **Returns**: the type, then always a description line, even if short
  (`The new beam.`). Methods that return a new object say `` ``self`` is unchanged.``
- **Raises**: `If ...` sentences, one per exception type.
- **See Also**: closely related public names, with a few words each.
- **Notes**: formulas and conventions that do not fit a parameter, kept short.
- **Examples**: only on main classes and non-obvious functions. Write values
  with `diffractix.units`, keep them runnable (they pass with
  `pytest --doctest-modules`), and print rounded or formatted numbers.

## Formatting

- Formulas as plain code text in double backticks: ``x + z * theta``,
  ``1/q = 1/R - i * wavelength / (pi * n * w**2)``. No LaTeX in docstrings.
- Parameter names in single backticks (`x`), code and literals in double (``inf``).
- Wrap at about 90 characters; keep CRLF line endings.
- Shared wording stays identical across classes, for example:
  - `Refractive index of the medium. Defaults to the system's ambient index.`
  - `Name used in results and messages. Defaults to an automatic name such as ``"Ray1"``.`
  - `Return the ... after a first-order (ABCD) transformation.`
