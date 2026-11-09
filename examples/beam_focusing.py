"""Focus a Gaussian laser beam onto a workpiece at a fixed position.

The optimizer moves a single lens and chooses its focal length so that the beam
waist lands on the workpiece with the requested spot radius. The workpiece stays
at the same absolute position because the working distance is defined as the
remaining system length. All lengths are in metres.
"""

from diffractix import Constraint, GaussianBeam, Plane, Solver, Space, System, ThinLens

WAVELENGTH = 1064e-9
INPUT_RADIUS = 0.75e-3
TARGET_WAIST_RADIUS = 30e-6
SYSTEM_LENGTH = 150e-3
MIN_WORKING_DISTANCE = 50e-3
LENS_CLEAR_DIAMETER = 4e-3
MAX_DIVERGENCE = 0.03  # Gaussian far-field half-angle in radians: paraxial limit
DEFOCUS = 2e-3         # Workpiece positioning error to inspect


# ------
# DESIGN
# ------
# The input beam has its waist at the source. The lens position and focal
# length start from deliberately poor values.
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)
target_focus = GaussianBeam.from_waist(w0=TARGET_WAIST_RADIUS, wavelength=WAVELENGTH).numeric()

input_gap = Space(d=60e-3, label="Source to lens")
input_gap.d.variable(bounds=(40e-3, None))
lens = ThinLens(f=45e-3, label="Focusing lens")
lens.f.variable(bounds=(35e-3, 110e-3))
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

# The spot must have the target radius, and the waist must sit on the workpiece.
solver.target(
    value=lambda sim: sim[workpiece].after.w,
    target=target_focus.w,
    scale=target_focus.w,
    label="Spot radius at workpiece",
)
solver.target(
    value=lambda sim: sim[workpiece].after.z0,
    target=0.0,
    scale=target_focus.z_r,
    label="Waist position at workpiece",
)

# Only the lens changes the beam's divergence, so checking after the lens keeps
# the whole system within the paraxial limit. The lens aperture keeps two beam
# radii of clearance; clipping and aberrations are not modelled.
solver.require(
    Constraint(lambda sim: sim[lens].after.divergence_angle,
               upper_bound=MAX_DIVERGENCE, label="Paraxial divergence after lens"),
    Constraint(lambda sim: sim[lens].after.w,
               upper_bound=LENS_CLEAR_DIAMETER / 4, label="Lens aperture clearance"),
)

solution = solver.solve(method="SLSQP")
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
      f"cost={solution.cost:.3e}, iterations={solution.iterations}, "
      f"evaluations={solution.function_evaluations}")
print(f"  {solution.message}")

table("Design")
row("Source to lens", input_gap.d.value, solution.value(input_gap.d), 1e3, "mm")
row("Lens focal length", lens.f.value, solution.value(lens.f), 1e3, "mm")
row("Working distance", working_distance.d.value, solution.value(working_distance.d), 1e3, "mm")
row("System length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Minimum working distance: {MIN_WORKING_DISTANCE * 1e3:.1f} mm")

before, after = initial[workpiece].after, optimized[workpiece].after
table("Focus at the workpiece")
row("Spot radius", before.w, after.w, 1e6, "um")
row("Waist offset", before.z0, after.z0, 1e3, "mm")
row("Waist radius", before.w0, after.w0, 1e6, "um")
row("Confocal parameter", before.b, after.b, 1e3, "mm")
row(f"Spot {DEFOCUS * 1e3:.0f} mm before", before.w_at_z(-DEFOCUS), after.w_at_z(-DEFOCUS), 1e6, "um")
row(f"Spot {DEFOCUS * 1e3:.0f} mm after", before.w_at_z(DEFOCUS), after.w_at_z(DEFOCUS), 1e6, "um")
print(f"Target waist radius: {target_focus.w * 1e6:.1f} um")

table("Model checks")
row("Beam radius at lens", initial[lens].after.w, optimized[lens].after.w, 1e3, "mm")
row("Divergence after lens", initial[lens].after.divergence_angle,
    optimized[lens].after.divergence_angle, unit="rad")
print(f"Limits: radius {LENS_CLEAR_DIAMETER / 4 * 1e3:.2f} mm, divergence {MAX_DIVERGENCE} rad")

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
