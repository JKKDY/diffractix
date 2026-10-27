# Diffractix

**Differentiable paraxial optics in Python.**

Diffractix is a Python library for modeling, simulating, and optimizing paraxial optical systems using ABCD matrices.

Optical systems are defined declaratively through optical elements, input states, and symbolic parameters. These descriptions are compiled into differentiable numerical simulations, enabling both forward propagation and inverse optical design.

## Features

- ABCD matrix-based optical propagation
- Ray, ray bundle, and Gaussian beam simulation
- Parameterized optical elements and system layouts
- Optimizable optical elements and input conditions
- Automatic differentiation using Autograd or JAX
- Constraints and numerical optimization for inverse design
- Extensible optical elements and input types

## Installation

Install the released package with:

```bash
pip install diffractix
```

The default installation includes Autograd, NumPy, SciPy, and Matplotlib. JAX is opt-in
with `pip install 'diffractix[jax]'`. If JAX is installed, automatic backend
selection prefers it; `DIFFRACTIX_BACKEND=autograd` explicitly selects Autograd.

To install from a local checkout:

```bash
pip install -e .
```

## Quick start

Diffractix lets you define optical systems declaratively and optimize their parameters. In this example, we adjust a lens's focal length to focus a ray onto an image plane.

```python
from diffractix import Ray, ThinLens, Plane, System, Solver

# Define the input ray and an adjustable lens
ray = Ray(x=0.01, theta=0.0)
lens = ThinLens(f=0.2)
lens.f.variable(bounds=(0.05, 0.5))

# Place elements at absolute positions (meters)
image_plane = Plane(label="Image plane")

system = System(input=ray)
system.add(lens, z=0.1)
system.add(image_plane, z=0.2)

simulation = system.build()
initial = simulation.run()

# Focus the ray onto the optical axis at the image plane
solver = Solver(simulation)
solver.target(
    value=lambda ctx: ctx.at(image_plane).x,
    target=0.0,
    scale=0.01,
)

solution = solver.solve(method="SLSQP")
optimized = solution.run()

print(f"Focal length: {lens.f.value * 1e3:.1f} → {solution[lens.f] * 1e3:.1f} mm")
print(f"Image height: {float(initial.at(image_plane).x) * 1e3:.1f} → {float(optimized.at(image_plane).x) * 1e3:.1f} mm")
```

Expected output:

```text
Focal length: 200.0 → 100.0 mm
Image height: 5.0 → 0.0 mm
```

The system automatically inserts propagation spaces between positioned elements. The solver then finds the focal length that brings the ray to the optical axis at the image plane.

## Project status

**Pre-release — targeting v0.0.1**

Diffractix is in active development. The API and internal architecture may change before the first stable release.

Documentation, examples, and visualization tools are being developed.

## License

See [LICENSE](https://github.com/JKKDY/diffractix/blob/main/LICENSE), included unchanged in both distributions.

## Development and releases

See [DEVELOPMENT.md](https://github.com/JKKDY/diffractix/blob/main/DEVELOPMENT.md) for local validation commands, the CI matrix,
package verification, and the PyPI Trusted Publishing setup.
