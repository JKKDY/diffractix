# Diffractix

**Differentiable paraxial optics in Python.**

Diffractix models, simulates and optimizes paraxial optical systems with ABCD matrices.
You describe a system from elements, a light source and parameters. Diffractix turns
that description into a differentiable simulation, which you can run to see how light
travels through the system, or hand to an optimizer to find the parameter values that
meet your requirements.

## Installation

```bash
pip install diffractix
```

The base install uses Autograd for derivatives and needs Python 3.11 or newer. Extras
add the JAX backend (`diffractix[jax]`) and the IPOPT and NLopt optimizers
(`diffractix[ipopt]`, `diffractix[nlopt]`, or both with `diffractix[all-solvers]`).

## Quick start

Choose the focal length of a lens so that an off-axis ray crosses the axis at an image
plane:

```python
from diffractix import Plane, Ray, Solver, System, ThinLens
from diffractix.units import mm

ray = Ray(x=10 * mm, theta=0.0)
lens = ThinLens(f=200 * mm).variable(f=(50 * mm, 500 * mm))
image_plane = Plane(label="Image plane")

system = System(source=ray)
system.add(lens, z=100 * mm)
system.add(image_plane, z=200 * mm)
simulation = system.build()

solver = Solver(simulation)
solver.target(value=lambda sim: sim[image_plane].after.x, target=0.0, scale=10 * mm)
solution = solver.solve()

print(f"Focal length: {solution.value(lens.f) / mm:.1f} mm")
```

```text title="Output"
Focal length: 100.0 mm
```

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
