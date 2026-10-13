from diffractix import (
    GaussianBeam,
    Objective,
    Plane,
    Solver,
    Space,
    System,
    ThinLens
)


# Configuration
WAVELENGTH = 1550e-9
PACKAGE_LENGTH = 25e-3
MIN_GAP = 4e-3


# Source and target fiber modes
source = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=10.4e-6)
target_mode = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=9.2e-6)


# Optical elements
input_gap = Space(d=5e-3, label="Input gap")
input_gap.d.variable(lower_bound=4e-3, upper_bound=10e-3)

lens1 = ThinLens(f=8e-3, label="Lens 1")

middle_gap = Space(d=8e-3, label="Lens spacing")
middle_gap.d.variable(
    lower_bound=5e-3,
    upper_bound=14e-3,
)

lens2 = ThinLens(f=7e-3, label="Lens 2")

# Derived parameter: total package length stays fixed.
output_gap = Space(
    d=PACKAGE_LENGTH - input_gap.d - middle_gap.d,
    label="Output gap",
)

target = Plane(label="Target fiber")


# Build optical system
system = System()
system.add_input_beam(source)
system.add(
    input_gap,
    lens1,
    middle_gap,
    lens2,
    output_gap,
    target,
)
# Minimum clearance before the output fiber
system.require(output_gap.d >= MIN_GAP)


# Compile and evaluate initial design
simulation = system.build()

initial_beam = simulation.run().at(target)
initial_overlap = initial_beam.overlap_with(target_mode)

print(f"Initial fiber overlap: {100 * initial_overlap:.6f} %")


# Inverse design
solver = Solver(simulation)

solver.target(
    Objective(
        lambda ctx: ctx.at(target).w / target_mode.w - 1,
        label="Target mode radius",
    ),
    Objective(
        lambda ctx: ctx.at(target).z0 / target_mode.z_r,
        label="Target waist position",
    ),
)



solution = solver.solve(method="trust-constr")


# Evaluate optimized design
matched_beam = solution.run().at(target)
overlap = matched_beam.overlap_with(target_mode)

print()
print("Optimization success:", solution.success)

print()
print(f"Input gap:     {1e3 * solution[input_gap.d]:.6f} mm")
print(f"Lens spacing:  {1e3 * solution[middle_gap.d]:.6f} mm")
print(f"Output gap:    {1e3 * solution[output_gap.d]:.6f} mm")

print()
print(f"Target radius: {1e6 * target_mode.w:.6f} um")
print(f"Matched radius:{1e6 * matched_beam.w:.6f} um")
print(f"Waist offset:  {1e6 * matched_beam.z0:.6f} um")
print(f"Fiber overlap: {100 * overlap:.8f} %")

print()
print("Objectives")
for result in solution.targets:
    print(
        f"  {result.objective.label}: "
        f"value={result.value:.6g}, "
        f"cost={result.cost:.6g}"
    )