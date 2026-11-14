"""Focus a Gaussian laser beam onto a workpiece at a fixed position.

The optimizer moves a single lens and chooses its focal length so that the beam
waist lands on the workpiece with the requested spot radius. The workpiece stays
at the same absolute position because the working distance is defined as the
remaining system length. Values are written with unit constants; diffractix
computes in SI.
"""

from diffractix import GaussianBeam, Plane, Solver, Space, System, ThinLens, units
from diffractix.units import mm, mrad, nm, um

WAVELENGTH = 1064 * nm
INPUT_RADIUS = 0.75 * mm
TARGET_WAIST_RADIUS = 30 * um
SYSTEM_LENGTH = 150 * mm
MIN_WORKING_DISTANCE = 50 * mm
LENS_CLEAR_DIAMETER = 4 * mm
MAX_DIVERGENCE = 30 * mrad  # Gaussian far-field half-angle: paraxial limit
DEFOCUS = 2 * mm            # Workpiece positioning error to inspect


# ------
# DESIGN
# ------
# The input beam has its waist at the source. The lens position and focal
# length start from deliberately poor values.
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)
target_focus = GaussianBeam.from_waist(w0=TARGET_WAIST_RADIUS, wavelength=WAVELENGTH).numeric()

input_gap = Space(d=60 * mm, label="Source to lens").variable(d=(40 * mm, None))
lens = ThinLens(f=45 * mm, label="Focusing lens").variable(f=(35 * mm, 110 * mm))
working_distance = Space(d=SYSTEM_LENGTH - input_gap.d, label="Working distance")
workpiece = Plane(label="Workpiece")

system = System(input_gap, lens, working_distance, workpiece, optical_input=source)
system.require(working_distance.d >= MIN_WORKING_DISTANCE)

simulation = system.build()
initial = simulation.run()


# ------------
# OPTIMIZATION
# ------------
solver = Solver(simulation)
sim = simulation.symbolic  # the result as expressions, for objectives and constraints

# The spot must have the target radius (a relative error, since the default scale of a
# target is its own size), and the waist must sit on the workpiece, within one
# Rayleigh range of scale.
solver.target(value=sim[workpiece].after.w, target=target_focus.w, label="Spot radius at workpiece")
solver.target(value=sim[workpiece].after.z0, target=0.0, scale=target_focus.z_r,
              label="Waist position at workpiece")

# Only the lens changes the beam's divergence, so checking after the lens keeps
# the whole system within the paraxial limit. The lens aperture keeps two beam
# radii of clearance; clipping and aberrations are not modelled.
solver.require(sim[lens].after.divergence_angle <= MAX_DIVERGENCE, label="Paraxial divergence after lens")
solver.require(sim[lens].after.w <= LENS_CLEAR_DIAMETER / 4, label="Lens aperture clearance")

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
row("Source to lens", input_gap.d.value, solution.value(input_gap.d), mm)
row("Lens focal length", lens.f.value, solution.value(lens.f), mm)
row("Working distance", working_distance.d.value, solution.value(working_distance.d), mm)
row("System length", initial.z[-1], optimized.z[-1], mm)
print(f"Minimum working distance: {units.format(MIN_WORKING_DISTANCE, mm)}")

before, after = initial[workpiece].after, optimized[workpiece].after
table("Focus at the workpiece")
row("Spot radius", before.w, after.w, um)
row("Waist offset", before.z0, after.z0, mm)
row("Waist radius", before.w0, after.w0, um)
row("Confocal parameter", before.b, after.b, mm)
row(f"Spot {units.format(DEFOCUS, mm)} before", before.w_at_z(-DEFOCUS), after.w_at_z(-DEFOCUS), um)
row(f"Spot {units.format(DEFOCUS, mm)} after", before.w_at_z(DEFOCUS), after.w_at_z(DEFOCUS), um)
print(f"Target waist radius: {units.format(target_focus.w, um)}")

table("Model checks")
row("Beam radius at lens", initial[lens].after.w, optimized[lens].after.w, mm)
row("Divergence after lens", initial[lens].after.divergence_angle,
    optimized[lens].after.divergence_angle, mrad)
print(f"Limits: radius {units.format(LENS_CLEAR_DIAMETER / 4, mm)}, "
      f"divergence {units.format(MAX_DIVERGENCE, mrad)}")

print("\nObjectives at the optimum")
for objective in solution.objectives:
    print(f"  {objective.label}: residual={objective.value:.3e}, cost={objective.cost:.3e}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    figure.suptitle("Beam focusing")
    initial.plot(ax=axes[0])
    optimized.plot(ax=axes[1])
    axes[0].set_title("Before optimization", loc="left")
    axes[1].set_title("After optimization", loc="left")
    axes[0].set_xlabel("")
    plt.show()
