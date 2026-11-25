# Diffractix

Trace rays and Gaussian beams through an optical system in Python. See where
the light lands, then use an optimizer to find lens positions, focal lengths,
or other values that meet your design goals.

Diffractix uses a single optical axis and the small-angle approximation.
The [optical model](guide/overview.md#the-optical-model) explains how it
represents light and optical elements.

## Installation

Diffractix needs Python 3.11 or newer.

```bash
pip install diffractix
```

## Trace a ray { #quick-start }

Start with a ray 10 mm above the optical axis and parallel to it. Put a lens
at 100 mm and an image plane at 200 mm. Diffractix fills the gaps between
these positions with spaces when you build the system.

```python
from diffractix import Plane, Ray, System, ThinLens
from diffractix.units import mm

ray = Ray(x=10 * mm, theta=0.0)
lens = ThinLens(f=200 * mm).variable(f=(50 * mm, 500 * mm))
image_plane = Plane(label="Image plane")

system = System(source=ray)
system.add(lens, z=100 * mm)
system.add(image_plane, z=200 * mm)
simulation = system.build()

result = simulation.run()
print(f"Image height: {result[image_plane].after.x / mm:.1f} mm")
```

```text title="Output"
Image height: 5.0 mm
```

`result[image_plane].after.x` gives the ray height after the image plane.
The ray is still above the axis.

## Choose a focal length

The lens starts at 200 mm. Marking it variable before the build lets the solver
choose a focal length between 50 and 500 mm. Set a target of zero for the ray
height at the image plane. A scale of 10 mm gives a residual of one in magnitude
when the ray misses the target by 10 mm. Use `simulation.symbolic` to refer to
the ray height as the solver varies the focal length.

```python
from diffractix import Solver

sim = simulation.symbolic
solver = Solver(simulation)
solver.target(value=sim[image_plane].after.x, target=0.0, scale=10 * mm)
solution = solver.solve()

print(f"Focal length: {solution.value(lens.f) / mm:.1f} mm")
```

```text title="Output"
Focal length: 100.0 mm
```

The solved focal length equals the 100 mm distance from the lens to the image plane.
[Focus a ray](examples/focus-a-ray.md) shows the ray path before and after the solve.

## Where to go next

- [How Diffractix works](guide/overview.md) explains the whole workflow on one page.
  Start here.
- The guide then goes through each step in detail:
  [declaring a system](guide/declaring.md),
  [parameters and expressions](guide/parameters.md),
  [building](guide/building.md),
  [running and reading results](guide/results.md),
  [solving](guide/solving.md) and
  [backends](guide/backends.md).
- The [API reference](api/index.md) documents every public class and function.
- The [examples](examples/index.md) are complete
  designs: beam focusing, a beam expander, fiber mode matching and more.

## Optional packages

The base install uses Autograd for derivatives and NLopt for optimization.
Install `diffractix[jax]` or `diffractix[casadi]` for an optional backend, and
`diffractix[plot]` for plotting. Install `diffractix[scipy]` for SciPy's
optimizers and HiGHS affine analysis, or `diffractix[ipopt]` for IPOPT.
`diffractix[all-solvers]` installs both extras.
The [backends guide](guide/backends.md) explains how to choose a numerical backend.
