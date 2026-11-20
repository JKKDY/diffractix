# Parameters and expressions

Every number in a design, such as a focal length, a distance or a ray's height, is a
**parameter**. Parameters are what the optimizer can change. Arithmetic on them builds
**expressions**, and comparisons build **constraints**. Together they form a graph that
the [build](building.md) turns into a function of the variable parameters.

## Fixed and variable parameters

A parameter is fixed unless you mark it variable. An element's parameters are its
attributes; mark them variable on the element, by name, with optional bounds:

```python
from diffractix import Parameter, Ray, Space, System, ThinLens
from diffractix.units import mm

lens = ThinLens(f=100 * mm).variable(f=(50 * mm, 200 * mm))
print(lens.f.value, lens.f.is_variable, lens.f.bounds)
```

```text title="Output"
0.1 True (0.05, 0.2)
```

`lens.variable("f")` marks `f` variable without changing its bounds,
`lens.f.variable(bounds=(...))` does the same through the parameter itself, and
`lens.fixed("f")` fixes it again. Bounds are a `(lower, upper)` pair; a `None` end keeps
the current bound, which is unbounded by default. The starting value must lie within
the bounds.

Light states have parameters too: a ray's `x`, `theta` and `n`, and a Gaussian beam's
wavelength and index can be variable. A Gaussian beam's `q` and a bundle's arrays of
heights and angles are plain numbers and stay fixed.

## Shared parameters

To use one value in several places, create a `Parameter` and pass it where a number is
expected. It is one design variable, wherever it appears:

```python
gap = Parameter(20 * mm, name="gap").variable(bounds=(5 * mm, 50 * mm))
system = System(Space(d=gap), ThinLens(f=100 * mm), Space(d=gap),
                source=Ray(x=1 * mm, theta=0.0))
print(len(system.build().parameter_data))
```

```text title="Output"
1
```

## Expressions

Arithmetic on parameters gives an expression: a value that is computed from other
values. An expression can be passed wherever a number is expected:

```python
distance = Space(d=2 * gap + 10 * mm)
print(distance.d)
print(distance.d.evaluate())
```

```text title="Output"
((2 * gap=0.02[V]) + 0.01)
0.05
```

The printed form shows each parameter with its current value, marked `[V]` if it is
variable and `[F]` if it is fixed. The distance is not a design variable of its own:
it follows `gap`. This is how you express relations in a design, such as a total
length that must stay constant, or two lenses that must have the same focal length.

Expressions support the arithmetic operators, `abs`, and methods such as `sqrt()`,
`exp()`, `sin()`, `maximum()` and `minimum()`. NumPy functions such as `np.sqrt` and
`np.abs` work on them as well.

An expression has no truth value, because its value is not known until the design is
evaluated: `if lens.f > 0:` raises a `SymbolicControlFlowError`. Use
`expression.evaluate()` to compute its current value, or turn the condition into a
requirement.

## Requirements

A comparison of expressions is a constraint. Attach it to the system or to an element
with `require`, so it becomes part of the design:

```python
lens = ThinLens(f=100 * mm, label="Lens").variable(f=(50 * mm, 200 * mm))
lens.require(lens.f >= 2 * gap, label="Focal length")
```

The comparisons `<=`, `>=` and `==` are supported; give each side of a range as its
own comparison, since a chained comparison such as `a < x < b` is not. Requirements on
results, such as a maximum beam radius at some element, are added to the solver
instead; see [Solving](solving.md).

The build checks every requirement. One that depends on a variable parameter is
handed to the optimizer, and may be violated by the starting values. One that does not
is checked once: if it holds it is dropped, and if it does not the build fails with an
error that names it. A design with a contradiction in its fixed values therefore never
reaches the optimizer.

Some requirements are added for you. Every `Space` requires `d >= 0`, every source
requires a positive refractive index (and a Gaussian beam a positive wavelength), and
generated gaps must not be negative.

## Values provided by the system

A few values are not set on an element but provided by the system at build: the
ambient refractive index, and the source's wavelength, which a `GaussianAperture` uses.
These are `Symbol`s, placeholders that the build replaces with the system's values.
You rarely need them directly; `System.bind` sets a value for your own symbols.
