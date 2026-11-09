"""Design a 4x Galilean expander for a Gaussian laser beam.

A diverging lens followed by a collimating lens enlarges the beam radius four
times and reduces its divergence four times. The optimizer chooses the lens
spacing and the second focal length, subject to a length limit and lens
apertures. All lengths are in metres.
"""

from time import perf_counter

from diffractix import Constraint, GaussianBeam, Plane, Solver, Space, System, ThinLens

WAVELENGTH = 1064e-9
INPUT_RADIUS = 0.5e-3
EXPANSION = 4.0
OUTPUT_DRIFT = 10e-3
MAX_OPTICAL_LENGTH = 100e-3

# Illustrative clear apertures. We keep the beam radius below a quarter of each
# diameter (two beam radii of clearance); clipping itself is not modelled.
LENS_DIAMETERS = {"Diverging lens": 4e-3, "Collimating lens": 10e-3}


# ------
# DESIGN
# ------
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)

# Deliberately poor starting values, still inside every limit.
lens1 = ThinLens(f=-20e-3, label="Diverging lens")
spacing = Space(d=80e-3, label="Lens spacing")
spacing.d.variable(bounds=(60e-3, 90e-3))
lens2 = ThinLens(f=100e-3, label="Collimating lens")
lens2.f.variable(bounds=(75e-3, 125e-3))
output = Plane(label="Output")

system = System(
    lens1,
    spacing,
    lens2,
    Space(d=OUTPUT_DRIFT, label="Output drift"),
    output,
    optical_input=source,
)
optical_length = sum(element.element_length for element in system.elements)
system.require(optical_length <= MAX_OPTICAL_LENGTH)

simulation = system.build()
initial = simulation.run()

input_beam = initial.source
target_radius = EXPANSION * input_beam.w
target_divergence = input_beam.divergence_angle / EXPANSION


# ------------
# OPTIMIZATION
# ------------
solver = Solver(simulation)

# Matching the radius alone would not make the expanded beam collimated.
solver.target(
    value=lambda sim: sim[output].after.w,
    target=target_radius,
    scale=target_radius,
    label="Output beam radius",
)
solver.target(
    value=lambda sim: sim[output].after.divergence_angle,
    target=target_divergence,
    scale=target_divergence,
    label="Output divergence",
)

for lens in (lens1, lens2):
    solver.require(Constraint(
        lambda sim, lens=lens: sim[lens].after.w,
        upper_bound=LENS_DIAMETERS[lens.label] / 4,
        label=f"{lens.label} aperture clearance",
    ))

start = perf_counter()
solution = solver.solve(method="SLSQP")
elapsed = perf_counter() - start
optimized = solution.run()


# -------
# RESULTS
# -------
def table(title):
    print(f"\n{title}")
    print(f"{'':26} {'Initial':>12} {'Optimized':>12}")
    print("-" * 56)


def row(label, before, after, scale=1.0, unit=""):
    print(f"{label:<26} {before * scale:>12.4f} {after * scale:>12.4f} {unit}")


print(f"Optimization: success={solution.success}, feasible={solution.feasible}, "
      f"cost={solution.cost:.3e}, time={elapsed:.2f} s")
print(f"  {solution.message}")

table("Design")
row("Lens spacing", spacing.d.value, solution.value(spacing.d), 1e3, "mm")
row("Collimating focal length", lens2.f.value, solution.value(lens2.f), 1e3, "mm")
row("Optical length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Limit: {MAX_OPTICAL_LENGTH * 1e3:.0f} mm")

before, after = initial[output].after, optimized[output].after
table("Output beam")
row("Radius", before.w, after.w, 1e3, "mm")
row("Expansion factor", before.w / input_beam.w, after.w / input_beam.w, unit="x")
row("Divergence", before.divergence_angle, after.divergence_angle, 1e3, "mrad")
print(f"Targets: {target_radius * 1e3:.4f} mm, {target_divergence * 1e3:.4f} mrad")

table("Beam radius at each lens")
for lens in (lens1, lens2):
    row(lens.label, initial[lens].after.w, optimized[lens].after.w, 1e3, "mm")
    print(f"  limit {LENS_DIAMETERS[lens.label] / 4 * 1e3:.3f} mm")

print("\nObjectives at the optimum")
for objective in solution.objectives:
    print(f"  {objective.label}: residual={objective.value:.3e}, cost={objective.cost:.3e}")
