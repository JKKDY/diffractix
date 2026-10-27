"""README quick start, respecting DIFFRACTIX_BACKEND when configured."""

# use autograd backend as that is usually faster for small problems
from diffractix import backend
backend.configure("autograd")

from diffractix import Ray, ThinLens, Plane, System, Solver

# Define the input ray and an adjustable lens
ray = Ray(x=0.01, theta=0.0)
lens = ThinLens(f=0.2)
lens.f.variable(bounds=(0.05, 0.5))

# Place elements at absolute positions (meters)
image_plane = Plane()

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

solution = solver.solve()
optimized = solution.run()

print(f"Focal length: {lens.f.value * 1e3:.1f} → {solution[lens.f] * 1e3:.1f} mm")
print(f"Image height: {float(initial.at(image_plane).x) * 1e3:.1f} → {float(optimized.at(image_plane).x) * 1e3:.1f} mm")
