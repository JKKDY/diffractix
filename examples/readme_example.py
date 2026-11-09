"""README quick start: choose a lens focal length that focuses a ray onto the axis.

All lengths are in metres.
"""

from diffractix import Plane, Ray, Solver, System, ThinLens

# An off-axis ray, parallel to the axis, and a lens whose focal length may vary.
ray = Ray(x=0.01, theta=0.0)
lens = ThinLens(f=0.2)
lens.f.variable(bounds=(0.05, 0.5))
image_plane = Plane(label="Image plane")

# Place the elements at absolute positions; the gaps become Spaces automatically.
system = System(optical_input=ray)
system.add(lens, z=0.1)
system.add(image_plane, z=0.2)

simulation = system.build()
initial = simulation.run()

# Ask for zero ray height at the image plane.
solver = Solver(simulation)
solver.target(
    value=lambda sim: sim[image_plane].after.x,
    target=0.0,
    scale=0.01,
)
solution = solver.solve()
optimized = solution.run()

print(f"Focal length: {lens.f.value * 1e3:.1f} → {solution.value(lens.f) * 1e3:.1f} mm")
print(f"Image height: {initial[image_plane].after.x * 1e3:.1f} → "
      f"{optimized[image_plane].after.x * 1e3:.1f} mm")
