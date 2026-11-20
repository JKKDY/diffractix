# Declaring a system

A design consists of a **source**, the light entering the system, and a sequence of
**elements** along the optical axis. A `System` collects them, together with the
requirements the design must meet. Declaring computes nothing: it only describes the
design, which [`build()`](building.md) later checks and compiles.

## Sources

The source is the light at the start of the system, `z = 0`. Diffractix has three kinds
of light state:

| State | Describes | Typical construction |
| --- | --- | --- |
| `Ray` | one ray: height `x` [m] and angle `theta` [rad] | `Ray(x=1 * mm, theta=0.0)` |
| `RayBundle` | many rays traced together | `RayBundle.parallel(radius=2 * mm, count=11)` |
| `GaussianBeam` | a fundamental Gaussian beam, by its complex beam parameter `q` | `GaussianBeam.from_waist(w0=0.5 * mm, wavelength=633 * nm)` |

A Gaussian beam can also be created from its radius and curvature
(`GaussianBeam.from_w_and_R`), its divergence (`from_divergence`) or a fiber's mode
(`from_fiber_tip`). A ray bundle can be parallel (`RayBundle.parallel`), converging
(`RayBundle.converging`), or built from arrays of heights and angles.

A system has exactly one source, passed as `System(source=...)` or set later with
`system.set_source(...)`.

## Elements

An element changes the light by its ray-transfer (ABCD) matrix, and may have a length.
All elements take keyword arguments and an optional `label`, which names them in
printed summaries and error messages.

| Element | Effect |
| --- | --- |
| `Space(d=...)` | free propagation over a distance `d` |
| `ThinLens(f=...)` | an ideal thin lens with focal length `f` |
| `Mirror(R=...)` | a flat or curved mirror; the path is unfolded, so light keeps travelling in `+z` |
| `Interface(n1=..., n2=..., R=...)` | refraction from index `n1` into `n2` at a flat or curved surface |
| `GRIN(d=..., g=..., n=...)` | a gradient-index medium |
| `GaussianAperture(a=...)` | a soft aperture with Gaussian transmission |
| `Plane()` | a named position that leaves the light unchanged, useful to read the light there |
| `ABCD(A=..., B=..., C=..., D=...)` | any element given by its matrix, also from an array with `ABCD.from_matrix` |

## Composites

A **composite** is a group of elements used as one: `Slab` (a block of material),
`ThickLens`, `Telescope` (two lenses, `f1 + f2` apart) and `FourF` (a 4f relay). Its
parts are available as attributes, so you can read the light at each one, or constrain
them:

```python
from diffractix import Ray, System, Telescope
from diffractix.units import mm

telescope = Telescope(f1=50 * mm, f2=100 * mm, label="Expander")
print(telescope.lens1.label, telescope.space.label, telescope.lens2.label)
```

```text title="Output"
Expander_L1 Expander_Drift Expander_L2
```

The build replaces a composite by its parts, but results can be looked up for the
composite's parts as for any other element. To write your own composite, subclass
`CompositeElement`; its API page shows how.

## Arranging elements

Elements are placed one after another along the axis in the order you give them. You
can do this in two ways, and mix them.

**In sequence.** Pass the elements in order, with explicit `Space` elements between
them. Each element starts where the previous one ends:

```python
from diffractix import Plane, Space, ThinLens

lens = ThinLens(f=100 * mm)
image = Plane(label="Image")
system = System(Space(d=50 * mm), lens, Space(d=100 * mm), image,
                source=Ray(x=1 * mm, theta=0.0))
```

**By position.** Add an element at an absolute position `z`. The build fills each gap
with a `Space`:

```python
system = System(source=Ray(x=1 * mm, theta=0.0))
system.add(lens, z=50 * mm)
system.add(image, z=150 * mm)
```

By default `z` is the position of the element's front. With `at="center"` or
`at="back"` it refers to the middle or the back instead; for a composite, `at` can also
be one of its parts. An element placed before the end of the previous one is an
error, and the build names the call that placed it.

The same element object may appear more than once, for example a lens passed twice in
a folded design. Each appearance is a separate stage of the simulation, and results
tell them apart by [occurrence](results.md#looking-up-the-light).

## Media and refractive index

The light always travels in some medium. Its refractive index starts at the source's
index, which defaults to the system's **ambient index** (`System(ambient_n=...)`,
default 1). An element without an index of its own inherits the medium it is in; a
`Space(d=...)` without `n` propagates in the current medium.

An `Interface` changes the medium from `n1` to `n2`. A `Slab` or `ThickLens` contains
its own interfaces, so it enters its material and returns to the surrounding medium
on its own. When an element declares an index that does not match the medium the light
is in, the build fails, or, if the indices are variable, adds a constraint that they
match.

## Requirements

A requirement is a constraint that is part of the design, such as a minimum distance
or a maximum focal length. Attach it to the system, or to an element:

```python
system.require(lens.f >= 80 * mm, label="Minimum focal length")
```

Requirements, and how they are checked, are covered in
[Parameters and expressions](parameters.md#requirements).

## Printing a design

`print(system)` shows the declared design: every placement, its parameters (marked as
variable, fixed or an expression) and the requirements. After the build,
`print(simulation)` shows the resolved layout with absolute positions and the
generated spaces.
