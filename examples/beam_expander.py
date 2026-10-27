"""Constrained design of a 4x Galilean Gaussian beam expander."""

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
INPUT_RADIUS = 0.5e-3
EXPANSION = 4.0
OUTPUT_DRIFT = 10e-3
MAX_OPTICAL_LENGTH = 100e-3

# Illustrative clear apertures for the two ideal thin lenses.
FIRST_LENS_DIAMETER = 4e-3
SECOND_LENS_DIAMETER = 10e-3


#------
# SETUP
#------
# Start with a Gaussian waist at the first lens.
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)

# Intentionally poor initial guess: short separation and insufficient
# positive focal length. Still inside the parameter and aperture limits.
lens1 = ThinLens(f=-20e-3, label="Diverging lens")

spacing = Space(d=80e-3, label="Lens spacing")
spacing.d.variable(bounds=(60e-3, 90e-3))

lens2 = ThinLens(f=100e-3, label="Collimating lens")
lens2.f.variable(bounds=(75e-3, 125e-3))

output_drift = Space(d=OUTPUT_DRIFT, label="Output drift")
output = Plane(label="Output")

system = System(
    lens1,
    spacing,
    lens2,
    output_drift,
    output,
    input=source,
)

optical_length = sum(element.element_length for element in system.elements)
system.require(optical_length <= MAX_OPTICAL_LENGTH)


#-----------
# SIMULATION
#-----------
simulation = system.build()

input_beam = simulation.source
target_radius = EXPANSION * input_beam.w
target_divergence = input_beam.divergence_angle / EXPANSION

# Keep the Gaussian 1/e^2 intensity radius inside half the clear-aperture
# radius (2w clearance). Aperture clipping is not simulated explicitly.
for lens, clear_diameter in (
    (lens1, FIRST_LENS_DIAMETER),
    (lens2, SECOND_LENS_DIAMETER),
):
    simulation.require(
        Constraint(
            lambda ctx, lens=lens: ctx.at(lens).w,
            upper_bound=clear_diameter / 4,
            label=f"{lens.label} aperture clearance",
        )
    )


#-------
# SOLVER
#-------
# Matching the radius alone does not ensure the expanded beam is collimated.
solver = Solver(simulation)
solver.target(
    value=lambda ctx: ctx.at(output).w,
    target=target_radius,
    scale=target_radius,
    label="Output beam radius",
)
solver.target(
    value=lambda ctx: ctx.at(output).divergence_angle,
    target=target_divergence,
    scale=target_divergence,
    label="Output divergence",
)

start = perf_counter()
solution = solver.solve(method="SLSQP")
elapsed = perf_counter() - start
initial = simulation.run()
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

# The current Solution API does not expose these diagnostics publicly.
backend_result = solution._result
print(f"Iterations:           {backend_result.iterations}")
print(f"Function evaluations: {backend_result.raw.nfev}")

print("\n\n###### DESIGN: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
for label, parameter in (
    ("Lens 1 focal length", lens1.f),
    ("Lens spacing", spacing.d),
    ("Lens 2 focal length", lens2.f),
):
    compare(label, parameter.value, solution[parameter], 1e3, "mm")
compare("Optical length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Maximum optical length: {1e3 * MAX_OPTICAL_LENGTH:.1f} mm")
print(f"Output drift: {1e3 * OUTPUT_DRIFT:.1f} mm (fixed)")

initial_beam = initial.at(output)
expanded_beam = optimized.at(output)

print("\n\n###### BEAM PERFORMANCE: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare("Output radius", initial_beam.w, expanded_beam.w, 1e3, "mm")
compare("Expansion factor", initial_beam.w / input_beam.w,
        expanded_beam.w / input_beam.w, unit="x")
compare("Output divergence", initial_beam.divergence_angle,
        expanded_beam.divergence_angle, 1e3, "mrad")
compare("Divergence reduction", input_beam.divergence_angle / initial_beam.divergence_angle,
        input_beam.divergence_angle / expanded_beam.divergence_angle, unit="x")
print(f"Target radius: {1e3 * target_radius:.4f} mm; "
      f"target divergence: {1e3 * target_divergence:.4f} mrad")

print("\n\n###### APERTURE CHECKS: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
for lens, clear_diameter in (
    (lens1, FIRST_LENS_DIAMETER),
    (lens2, SECOND_LENS_DIAMETER),
):
    compare(f"{lens.label} radius", initial.at(lens).w,
            optimized.at(lens).w, 1e3, "mm")
    print(f"  Beam-radius limit: {1e3 * clear_diameter / 4:.4f} mm")

print("\n\n###### OBJECTIVES AT OPTIMUM ######")
for result in solution.targets:
    print(f"  {result.objective.label}: "
          f"residual={result.value:.3e}, cost={result.cost:.3e}")
