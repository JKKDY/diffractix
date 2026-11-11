"""Design a 4x Galilean expander for a Gaussian laser beam.

A diverging lens followed by a collimating lens enlarges the beam radius four
times and reduces its divergence four times. The optimizer chooses the lens
spacing and the second focal length, subject to a length limit and lens
apertures. Values are written with unit constants; diffractix computes in SI.
"""

from diffractix import GaussianBeam, Plane, Solver, Space, System, ThinLens, units
from diffractix.units import mm, mrad, nm

WAVELENGTH = 1064 * nm
INPUT_RADIUS = 0.5 * mm
EXPANSION = 4.0
OUTPUT_DRIFT = 10 * mm
MAX_OPTICAL_LENGTH = 100 * mm

# Illustrative clear apertures. We keep the beam radius below a quarter of each
# diameter (two beam radii of clearance); clipping itself is not modelled.
LENS_DIAMETERS = {"Diverging lens": 4 * mm, "Collimating lens": 10 * mm}


# ------
# DESIGN
# ------
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)

# Deliberately poor starting values, still inside every limit.
lens1 = ThinLens(f=-20 * mm, label="Diverging lens")
spacing = Space(d=80 * mm, label="Lens spacing").variable(d=(60 * mm, 90 * mm))
lens2 = ThinLens(f=100 * mm, label="Collimating lens").variable(f=(75 * mm, 125 * mm))
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
sim = simulation.symbolic  # the result as expressions, for objectives and constraints

# Matching the radius alone would not make the expanded beam collimated. The default
# scale of a target is its own size, so both are relative errors.
solver.target(value=sim[output].after.w, target=target_radius, label="Output beam radius")
solver.target(value=sim[output].after.divergence_angle, target=target_divergence,
              label="Output divergence")

for lens in (lens1, lens2):
    solver.require(sim[lens].after.w <= LENS_DIAMETERS[lens.label] / 4,
                   label=f"{lens.label} aperture clearance")

solution = solver.solve(method="SLSQP")
optimized = solution.run()


# -------
# RESULTS
# -------
def table(title):
    print(f"\n{title}")
    print(f"{'':26} {'Initial':>12} {'Optimized':>12}")
    print("-" * 56)


def row(label, before, after, unit=None):
    """A row of values in unit (a unit constant), or as plain numbers without one."""
    scale, symbol = (unit, unit.symbol) if unit is not None else (1.0, "")
    print(f"{label:<26} {float(before) / scale:>12.4f} {float(after) / scale:>12.4f} {symbol}")


print(f"Optimization: success={solution.success}, feasible={solution.feasible}, "
      f"cost={solution.cost:.3e}, iterations={solution.iterations}, "
      f"evaluations={solution.function_evaluations}")
print(f"  {solution.message}")

table("Design")
row("Lens spacing", spacing.d.value, solution.value(spacing.d), mm)
row("Collimating focal length", lens2.f.value, solution.value(lens2.f), mm)
row("Optical length", initial.z[-1], optimized.z[-1], mm)
print(f"Limit: {units.format(MAX_OPTICAL_LENGTH, mm)}")

before, after = initial[output].after, optimized[output].after
table("Output beam")
row("Radius", before.w, after.w, mm)
row("Expansion factor", before.w / input_beam.w, after.w / input_beam.w)
row("Divergence", before.divergence_angle, after.divergence_angle, mrad)
print(f"Targets: {units.format(target_radius, mm)}, {units.format(target_divergence, mrad)}")

table("Beam radius at each lens")
for lens in (lens1, lens2):
    row(lens.label, initial[lens].after.w, optimized[lens].after.w, mm)
    print(f"  limit {units.format(LENS_DIAMETERS[lens.label] / 4, mm)}")

print("\nObjectives at the optimum")
for objective in solution.objectives:
    print(f"  {objective.label}: residual={objective.value:.3e}, cost={objective.cost:.3e}")
