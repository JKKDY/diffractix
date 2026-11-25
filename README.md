# Diffractix

[![codecov](https://codecov.io/gh/JKKDY/diffractix/graph/badge.svg?token=5Q03OMG1NX)](https://codecov.io/gh/JKKDY/diffractix)

**Differentiable paraxial optics in Python.**

Diffractix models, simulates and optimizes paraxial optical systems with ABCD matrices.
You declare a system from elements, a light source and parameters; Diffractix compiles it
into a differentiable simulation that you can run forward or hand to an optimizer for
inverse design.

## Features

- Rays, ray bundles and Gaussian beams propagated with ABCD matrices
- Thin and thick lenses, mirrors, interfaces, slabs, GRIN media, Gaussian apertures,
  telescopes and 4f systems, plus custom ABCD elements
- Any element parameter, position or source property can be a design variable
- Automatic differentiation with Autograd, JAX or CasADi
- Objectives and constraints for inverse design, solved with NLopt, SciPy, IPOPT or CasADi
- Plots of beams and rays through the system

## Installation

```bash
pip install diffractix
```

The base install uses Autograd, NumPy and NLopt and needs Python 3.11 or newer.

- `pip install "diffractix[jax]"` adds the JAX backend.
- `pip install "diffractix[casadi]"` adds the CasADi backend.
- `pip install "diffractix[plot]"` adds Matplotlib for plotting.
- `pip install "diffractix[scipy]"` adds the SciPy optimizers and HiGHS for affine
  constraint analysis.
- `pip install "diffractix[ipopt]"` adds IPOPT. Install both extras with
  `pip install "diffractix[all-solvers]"`. IPOPT is easiest to install with
  `conda install -c conda-forge cyipopt`.

Set `DIFFRACTIX_BACKEND` to `autograd`, `jax`, or `casadi` to select a backend.
By default Diffractix uses JAX when installed and Autograd otherwise.

`Solver.solve()` uses NLopt's SLSQP by default. Select an optimizer explicitly
to use its specific options.

## Quick start

Choose the focal length of a lens so that an off-axis ray crosses the axis at an image
plane:

```python
from diffractix import Plane, Ray, Solver, System, ThinLens
from diffractix.units import mm

# An off-axis ray, parallel to the axis, and a lens whose focal length may vary.
ray = Ray(x=10 * mm, theta=0.0)
lens = ThinLens(f=200 * mm).variable(f=(50 * mm, 500 * mm))
image_plane = Plane(label="Image plane")

# Place the elements at absolute positions; the gaps become Spaces automatically.
system = System(source=ray)
system.add(lens, z=100 * mm)
system.add(image_plane, z=200 * mm)

simulation = system.build()
initial = simulation.run()

# Ask for zero ray height at the image plane.
solver = Solver(simulation)
solver.target(
    value=lambda sim: sim[image_plane].after.x,
    target=0.0,
    scale=10 * mm,
)
solution = solver.solve()
optimized = solution.run()

print(f"Focal length: {lens.f.value / mm:.1f} -> {solution.value(lens.f) / mm:.1f} mm")
print(f"Image height: {initial[image_plane].after.x / mm:.1f} -> "
      f"{optimized[image_plane].after.x / mm:.1f} mm")
```

```text
Focal length: 200.0 -> 100.0 mm
Image height: 5.0 -> 0.0 mm
```

The workflow is always the same: declare a `System`, `build()` it into a `Simulation`,
`run()` it to get the light at every element, and give it to a `Solver` to optimize the
variables. More complete designs (beam expanders, fiber coupling, multi-station
delivery) are in the [examples](https://github.com/JKKDY/diffractix/tree/main/examples).

## Documentation

The documentation, including the API reference, is at
[jkkdy.github.io/diffractix](https://jkkdy.github.io/diffractix/).

## Project status

Version 0.1.0 is the first release. The API may still change in later 0.x releases.

## License

Diffractix is licensed under the GNU Affero General Public License v3.0 or later; see
[LICENSE](https://github.com/JKKDY/diffractix/blob/main/LICENSE).
