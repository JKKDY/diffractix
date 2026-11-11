"""README quick start: choose a lens focal length that focuses a ray onto the axis.

Values are written with unit constants; diffractix computes in SI (metres).
"""

from diffractix import Plane, Ray, Solver, System, ThinLens
from diffractix.units import mm

# An off-axis ray, parallel to the axis, and a lens whose focal length may vary.
ray = Ray(x=10 * mm, theta=0.0)
lens = ThinLens(f=200 * mm).variable(f=(50 * mm, 500 * mm))
image_plane = Plane(label="Image plane")

# Place the elements at absolute positions; the gaps become Spaces automatically.
system = System(optical_input=ray)
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

print(f"Focal length: {lens.f.value / mm:.1f} → {solution.value(lens.f) / mm:.1f} mm")
print(f"Image height: {initial[image_plane].after.x / mm:.1f} → "
      f"{optimized[image_plane].after.x / mm:.1f} mm")
