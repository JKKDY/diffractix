"""Design a Gaussian laser focusing system with a fixed workpiece position.

Before/after plots require the optional diffractix[plot] extra.
"""

from time import perf_counter

from diffractix import (
    Constraint,
    GaussianBeam,
    Plane,
    Solver,
    Space,
    System,
    ThinLens,
)

# All lengths are in meters.
WAVELENGTH = 1064e-9
INPUT_RADIUS = 0.75e-3
TARGET_WAIST_RADIUS = 30e-6
SYSTEM_LENGTH = 150e-3
MIN_WORKING_DISTANCE = 50e-3
LENS_CLEAR_DIAMETER = 4e-3
MAX_DIVERGENCE = 0.03  # Gaussian far-field half-angle, radians
DEFOCUS = 2e-3         # Workpiece positioning error to inspect


#------
# SETUP
#------
# The input beam is at a waist at the source plane. Its lens position and
# focal length are deliberately poor starting values.
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)
target_focus = GaussianBeam.from_waist(
    w0=TARGET_WAIST_RADIUS, wavelength=WAVELENGTH
).numeric()

input_gap = Space(d=60e-3, label="Source-to-lens distance")
input_gap.d.variable(bounds=(40e-3, None))

lens = ThinLens(f=45e-3, label="Focusing lens")
lens.f.variable(bounds=(35e-3, 110e-3))

# Keep the workpiece at the same physical position when the lens moves.
working_distance = Space(
    d=SYSTEM_LENGTH - input_gap.d,
    label="Working distance",
)
workpiece = Plane(label="Workpiece")

system = System(
    input_gap,
    lens,
    working_distance,
    workpiece,
    input=source,
)
system.require(working_distance.d >= MIN_WORKING_DISTANCE)


#-----------
# SIMULATION
#-----------
simulation = system.build()
initial = simulation.run()

# Check that the far-field Gaussian divergence stays within our chosen
# paraxial engineering limit throughout the system.
for i, info in enumerate(simulation.element_info, start=1):
    simulation.require(
        Constraint(
            lambda ctx, i=i: ctx.states[i].divergence_angle,
            upper_bound=MAX_DIVERGENCE,
            label=f"Paraxial divergence after {info.label or info.path or info.type_name}",
        )
    )

# Require a clear-aperture radius of at least twice the beam radius.
# Clipping and lens aberrations are not included in the ideal ABCD model.
simulation.require(
    Constraint(
        lambda ctx: ctx.at(lens).w,
        upper_bound=LENS_CLEAR_DIAMETER / 4,
        label="Focusing lens aperture clearance",
    )
)


#-------
# SOLVER
#-------
# The output beam must have the desired waist size, and the waist must
# coincide with the fixed workpiece plane.
solver = Solver(simulation)
solver.target(
    value=lambda ctx: ctx.at(workpiece).w,
    target=target_focus.w,
    scale=target_focus.w,
    label="Spot radius at workpiece",
)
solver.target(
    value=lambda ctx: ctx.at(workpiece).z0,
    target=0.0,
    scale=target_focus.z_r,
    label="Waist position at workpiece",
)

start = perf_counter()
solution = solver.solve(method="SLSQP", use_jit=True)
elapsed = perf_counter() - start
optimized = solution.run()


#--------
# READOUT
#--------
def compare(label, before, after, scale=1.0, unit=""):
    """Print one row of the initial-versus-optimized comparison."""
    print(f"{label:<23} {before * scale:>12.4f} {after * scale:>12.4f} {unit}")


print("###### OPTIMIZATION ######")
print(f"Success:              {solution.success}")
print(f"Feasible:             {solution.feasible}")
print(f"Termination:          {solution.message}")
print(f"Final objective:      {solution.cost:.3e}")
print(f"Solve time:           {elapsed:.3f} s")

# Iteration diagnostics are not yet exposed publicly by Solution.
backend_result = solution._result
print(f"Iterations:           {backend_result.iterations}")
print(f"Function evaluations: {backend_result.raw.nfev}")

print("\n\n###### DESIGN: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare("Source-to-lens distance", input_gap.d.value, solution[input_gap.d], 1e3, "mm")
compare("Lens focal length", lens.f.value, solution[lens.f], 1e3, "mm")
compare("Working distance", working_distance.d.value, solution[working_distance.d], 1e3, "mm")
compare("System length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Minimum working distance: {1e3 * MIN_WORKING_DISTANCE:.1f} mm")

initial_beam = initial.at(workpiece)
focused_beam = optimized.at(workpiece)

print("\n\n###### FOCUS AT WORKPIECE: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare("Spot radius", initial_beam.w, focused_beam.w, 1e6, "um")
compare("Waist offset", initial_beam.z0, focused_beam.z0, 1e3, "mm")
compare("Minimum waist radius", initial_beam.w0, focused_beam.w0, 1e6, "um")
compare("Confocal parameter", initial_beam.b, focused_beam.b, 1e3, "mm")
print(f"Target waist radius: {1e6 * target_focus.w:.4f} um")

print("\n\n###### DEFOCUS AND MODEL CHECKS: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare("Spot at -2 mm", initial_beam.w_at_z(-DEFOCUS),
        focused_beam.w_at_z(-DEFOCUS), 1e6, "um")
compare("Spot at +2 mm", initial_beam.w_at_z(DEFOCUS),
        focused_beam.w_at_z(DEFOCUS), 1e6, "um")
compare("Lens beam radius", initial.at(lens).w, optimized.at(lens).w, 1e3, "mm")
compare("Worst divergence", max(s.divergence_angle for s in initial.states),
        max(s.divergence_angle for s in optimized.states), unit="rad")
print(f"Lens beam-radius limit: {1e3 * LENS_CLEAR_DIAMETER / 4:.3f} mm")
print(f"Divergence limit: {MAX_DIVERGENCE:.3f} rad")

print("\n\n###### OBJECTIVES AT OPTIMUM ######")
for result in solution.targets:
    print(f"  {result.objective.label}: "
          f"residual={result.value:.3e}, cost={result.cost:.3e}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Beam focusing", fontsize=12)
    axes[0].set_title("Before optimization", loc="left", fontsize=11)
    axes[1].set_title("After optimization", loc="left", fontsize=11)
    initial.plot(ax=axes[0])
    optimized.plot(ax=axes[1])
    axes[0].set_xlabel("")
    plt.show()
