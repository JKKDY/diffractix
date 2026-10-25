"""Multi-station laser delivery with six lenses and refractive instrument modules.

Optimize six lens positions, three negative focal lengths, the radii and
thicknesses of three positive thick lenses, and the glass thicknesses of two
fixed-position instrument modules: 20 design variables in total.

The AOM/EOM slabs describe passive first-order refraction only, not modulation.
All lengths are in meters; named stations stay at fixed absolute positions.
"""

from time import perf_counter

from diffractix import (
    Constraint,
    GaussianBeam,
    Parameter,
    Plane,
    Slab,
    Solver,
    Space,
    System,
    ThickLens,
    ThinLens,
    Backend
)

from diffractix.backend import current_backend, xp
print(current_backend())

xp.add(1.0, 2.0)

print(current_backend())  # "jax" or "autograd"

WAVELENGTH = 1064e-9
INPUT_RADIUS = 0.65e-3

AOM_Z, AOM_EXIT_Z = 200e-3, 215e-3
EOM_Z, EOM_EXIT_Z = 430e-3, 450e-3
SCANNER_Z, WORKPIECE_Z = 625e-3, 850e-3

AOM_RADIUS, AOM_EXIT_RADIUS = 0.85e-3, 0.90e-3
EOM_RADIUS, EOM_EXIT_RADIUS = 1.10e-3, 1.20e-3
SCANNER_RADIUS, FOCUS_RADIUS = 1.32e-3, 45e-6

GLASS_INDEX = 1.52
AOM_CRYSTAL_INDEX = 1.45
EOM_CRYSTAL_INDEX = 1.62
MIN_WORKING_DISTANCE = 110e-3
MIN_LENS_GAP = 8e-3
MAX_DIVERGENCE = 0.03

THIN_LENS_DIAMETER = 9e-3
FIRST_LENS_DIAMETER = 5e-3
THICK_LENS_DIAMETER = 9e-3
STATION_DIAMETERS = (6e-3, 6e-3, 8e-3, 8e-3, 8e-3)

# SLSQP is a better first solver for this heavily constrained example.
# The two methods use different SciPy options.
SOLVER_METHOD = "SLSQP"  # Alternative: "trust-constr"
SOLVER_OPTIONS = {
    "SLSQP": {"maxiter": 650, "ftol": 1e-11, "disp": False},
    "trust-constr": {"maxiter": 350, "gtol": 1e-7, "xtol": 1e-9, "verbose": 0},
}


#------
# SETUP
#------
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)
target_focus = GaussianBeam.from_waist(w0=FOCUS_RADIUS, wavelength=WAVELENGTH)
if source.numeric().divergence_angle > MAX_DIVERGENCE:
    raise ValueError("Source exceeds the paraxial divergence limit.")

# Deliberately poor but feasible initial design, for testing convergence from
# a significantly mismatched optical layout. The initial workpiece spot is
# about 124 um instead of 45 um; its waist is roughly 17.8 mm off target.
# All 20 initial values remain within their bounds and meet the modeled
# mechanical, aperture, and paraxial requirements.
# Six independently adjustable lens *front-vertex* positions.
# Disjoint bounds preserve optical order across the fixed instrument stations.
z1 = Parameter(135e-3, name="Lens 1 position").variable(bounds=(40e-3, 170e-3))
z2 = Parameter(270e-3, name="Lens 2 position").variable(bounds=(240e-3, 330e-3))
z3 = Parameter(365e-3, name="Lens 3 position").variable(bounds=(345e-3, 400e-3))
z4 = Parameter(500e-3, name="Lens 4 position").variable(bounds=(475e-3, 585e-3))
z5 = Parameter(675e-3, name="Lens 5 position").variable(bounds=(655e-3, 705e-3))
z6 = Parameter(720e-3, name="Lens 6 position").variable(bounds=(715e-3, 730e-3))
positions = (z1, z2, z3, z4, z5, z6)

# Negative lenses remain ideal. Positive lenses have actual refracting surfaces,
# center thicknesses, and a fixed glass index. R1 > 0 and R2 < 0 describe
# biconvex shapes in the forward direction.
lens1 = ThinLens(f=-160e-3, label="Lens 1 (expander)")
lens1.f.variable(bounds=(-400e-3, -45e-3))

lens2 = ThickLens(d=6e-3, n=GLASS_INDEX, R1=150e-3, R2=-250e-3,
                  label="Lens 2 (thick relay)")
lens2.R1.variable(bounds=(90e-3, 550e-3))
lens2.R2.variable(bounds=(-550e-3, -90e-3))
lens2.d.variable(bounds=(3e-3, 11e-3))

lens3 = ThinLens(f=-340e-3, label="Lens 3 (expander)")
lens3.f.variable(bounds=(-500e-3, -45e-3))

lens4 = ThickLens(d=8e-3, n=GLASS_INDEX, R1=240e-3, R2=-280e-3,
                  label="Lens 4 (thick relay)")
lens4.R1.variable(bounds=(90e-3, 500e-3))
lens4.R2.variable(bounds=(-500e-3, -90e-3))
lens4.d.variable(bounds=(3e-3, 12e-3))

lens5 = ThinLens(f=-160e-3, label="Lens 5 (expander)")
lens5.f.variable(bounds=(-500e-3, -45e-3))

lens6 = ThickLens(d=4e-3, n=GLASS_INDEX, R1=180e-3, R2=-220e-3,
                  label="Lens 6 (thick focusing)")
lens6.R1.variable(bounds=(70e-3, 350e-3))
lens6.R2.variable(bounds=(-350e-3, -70e-3))
lens6.d.variable(bounds=(3e-3, 10e-3))

lenses = (lens1, lens2, lens3, lens4, lens5, lens6)
thin_lenses = (lens1, lens3, lens5)
thick_lenses = (lens2, lens4, lens6)

# Passive crystal slabs inside *fixed-length* AOM and EOM modules.
# Changing the crystal thickness changes the surrounding air allowances, not
# the entrance/exit station coordinates. Indices are fixed material properties.
aom_crystal = Slab(d=9e-3, n=AOM_CRYSTAL_INDEX, label="AOM crystal")
aom_crystal.d.variable(bounds=(6e-3, 14e-3))
eom_crystal = Slab(d=17e-3, n=EOM_CRYSTAL_INDEX, label="EOM crystal")
eom_crystal.d.variable(bounds=(8e-3, 18e-3))

aom = Plane(label="AOM")
aom_exit = Plane(label="AOM exit")
eom = Plane(label="EOM")
eom_exit = Plane(label="EOM exit")
scanner = Plane(label="Scanner")
workpiece = Plane(label="Workpiece")
stations = (
    (aom, AOM_RADIUS),
    (aom_exit, AOM_EXIT_RADIUS),
    (eom, EOM_RADIUS),
    (eom_exit, EOM_EXIT_RADIUS),
    (scanner, SCANNER_RADIUS),
)

# An absolute lens position refers to the front surface. Subtract each thick
# lens's physical length before the next absolute station or lens coordinate.
working_distance = Space(d=WORKPIECE_Z - z6 - lens6.d, label="Working distance")
system = System(
    Space(d=z1, label="Input gap"),
    lens1,
    Space(d=AOM_Z - z1, label="To AOM"),
    aom,
    Space(d=(AOM_EXIT_Z - AOM_Z - aom_crystal.d) / 2, label="AOM input allowance"),
    aom_crystal,
    Space(d=(AOM_EXIT_Z - AOM_Z - aom_crystal.d) / 2, label="AOM output allowance"),
    aom_exit,
    Space(d=z2 - AOM_EXIT_Z, label="To lens 2"),
    lens2,
    Space(d=z3 - z2 - lens2.d, label="Between lenses 2 and 3"),
    lens3,
    Space(d=EOM_Z - z3, label="To EOM"),
    eom,
    Space(d=(EOM_EXIT_Z - EOM_Z - eom_crystal.d) / 2, label="EOM input allowance"),
    eom_crystal,
    Space(d=(EOM_EXIT_Z - EOM_Z - eom_crystal.d) / 2, label="EOM output allowance"),
    eom_exit,
    Space(d=z4 - EOM_EXIT_Z, label="To lens 4"),
    lens4,
    Space(d=SCANNER_Z - z4 - lens4.d, label="To scanner"),
    scanner,
    Space(d=z5 - SCANNER_Z, label="To lens 5"),
    lens5,
    Space(d=z6 - z5, label="Between lenses 5 and 6"),
    lens6,
    working_distance,
    workpiece,
    input=source,
)
system.require(
    working_distance.d >= MIN_WORKING_DISTANCE,
    z3 - z2 - lens2.d >= MIN_LENS_GAP,
)


#-----------
# SIMULATION
#-----------
simulation = system.build()

# Only focusing elements change the Gaussian far-field divergence in air.
# The plane-parallel AOM/EOM slabs have no optical power: their plane faces
# change the local refractive index but do not add an independent air-side
# divergence constraint. Checking both faces duplicated existing limits and
# made the optimizer's constraint Jacobian unnecessarily rank-deficient.
# For a thick lens, check BOTH curved surfaces (including the glass side).
power_elements = (
    *thin_lenses,
    *(surface for lens in thick_lenses for surface in (lens.front, lens.back)),
)
for element in power_elements:
    simulation.require(
        Constraint(
            lambda ctx, element=element: ctx.after(element).divergence_angle,
            upper_bound=MAX_DIVERGENCE,
            label=f"Paraxial divergence after {element.label}",
        )
    )

# Each aperture limit is w <= diameter/4 (2w beam clearance). This is a
# paraxial validity/clearance requirement, not physical diffraction clipping.
# For the power-free slab modules, w^2 is convex along the optical path;
# checking the two station endpoints with the same aperture limit also bounds
# the beam at the internal plane interfaces. Avoid duplicate interior checks.
apertures = (
    (lens1, FIRST_LENS_DIAMETER),
    (lens3, THIN_LENS_DIAMETER),
    (lens5, THIN_LENS_DIAMETER),
    *((surface, THICK_LENS_DIAMETER)
      for lens in thick_lenses for surface in (lens.front, lens.back)),
    *zip((aom, aom_exit, eom, eom_exit, scanner), STATION_DIAMETERS),
)
for element, diameter in apertures:
    simulation.require(
        Constraint(
            lambda ctx, element=element: ctx.at(element).w,
            upper_bound=diameter / 4,
            label=f"{element.label} aperture clearance",
        )
    )


#-------
# SOLVER
#-------
solver = Solver(simulation)
for station, radius in stations:
    solver.target(
        value=lambda ctx, station=station: ctx.at(station).w,
        target=radius,
        scale=radius,
        label=f"{station.label} beam radius",
    )

solver.target(
    value=lambda ctx: ctx.at(workpiece).w,
    target=FOCUS_RADIUS,
    scale=FOCUS_RADIUS,
    label="Workpiece spot radius",
)
solver.target(
    value=lambda ctx: ctx.at(workpiece).z0,
    target=0.0,
    scale=target_focus.z_r,
    label="Workpiece waist position",
)

# Additional *soft* manufacturing preferences. These remove some of the
# otherwise arbitrary degrees of freedom without making the design impossible:
# prefer symmetric biconvex lenses, ~6 mm center thickness, and familiar
# crystal lengths. Hard bounds above still control manufacturable ranges.
for i, lens in ((2, lens2), (4, lens4), (6, lens6)):
    solver.target((lens.R1 + lens.R2) / 0.10, label=f"Lens {i} surface symmetry")
    solver.target((lens.d - 6e-3) / 12e-3, label=f"Lens {i} preferred thickness")
for label, slab, desired in (
    ("AOM", aom_crystal, 10e-3),
    ("EOM", eom_crystal, 14e-3),
):
    solver.target((slab.d - desired) / 20e-3, label=f"{label} preferred crystal thickness")

start = perf_counter()
solution = solver.solve(method=SOLVER_METHOD, options=SOLVER_OPTIONS[SOLVER_METHOD])
elapsed = perf_counter() - start
initial = simulation.run()
optimized = solution.run()


#--------
# READOUT
#--------
def compare(label, before, after, scale=1.0, unit=""):
    """Print an initial-versus-optimized comparison row."""
    print(f"{label:<28} {before * scale:>12.4f} {after * scale:>12.4f} {unit}")


def heading(title):
    print(f"\n\n###### {title} ######")
    print(f"{'':28} {'Initial':>12} {'Optimized':>12}")
    print("-" * 57)


print("###### OPTIMIZATION ######")
print(f"Method:               {SOLVER_METHOD}")
print(f"Success:              {solution.success}")
print(f"Feasible:             {solution.feasible}")
print(f"Termination:          {solution.message}")
print(f"Final objective:      {solution.cost:.3e}")
print(f"Solve time:           {elapsed:.3f} s")
backend_result = solution._result  # Private accessor: reporting only.
print(f"Iterations:           {backend_result.iterations}")
print(f"Function evaluations: {backend_result.raw.nfev}")
if hasattr(backend_result.raw, "optimality"):
    print(f"Optimality (scaled):  {backend_result.raw.optimality:.3e}")
if hasattr(backend_result.raw, "constr_violation"):
    print(f"Constraint violation (scaled): {backend_result.raw.constr_violation:.3e}")
print("Design variables:    20")

heading("DESIGN: BEFORE / AFTER")
for i, (pos, lens) in enumerate(zip(positions, lenses), start=1):
    compare(f"Lens {i} position", pos.value, solution[pos], 1e3, "mm")
    if isinstance(lens, ThinLens):
        compare(f"Lens {i} focal length", lens.f.value, solution[lens.f], 1e3, "mm")
    else:
        for name in ("R1", "R2", "d"):
            parameter = getattr(lens, name)
            compare(f"Lens {i} {name}", parameter.value, solution[parameter], 1e3, "mm")
compare("AOM crystal thickness", aom_crystal.d.value, solution[aom_crystal.d], 1e3, "mm")
compare("EOM crystal thickness", eom_crystal.d.value, solution[eom_crystal.d], 1e3, "mm")
compare("Working distance", WORKPIECE_Z - z6.value - lens6.d.value,
        solution[working_distance.d], 1e3, "mm")
compare("Total system length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Fixed stations: AOM {1e3 * AOM_Z:.0f}–{1e3 * AOM_EXIT_Z:.0f} mm; "
      f"EOM {1e3 * EOM_Z:.0f}–{1e3 * EOM_EXIT_Z:.0f} mm; "
      f"scanner {1e3 * SCANNER_Z:.0f} mm")
print(f"Minimum working distance: {1e3 * MIN_WORKING_DISTANCE:.0f} mm")
print(f"Thick-lens glass index: {GLASS_INDEX:.2f}; "
      f"AOM/EOM crystal indices: {AOM_CRYSTAL_INDEX:.2f}/{EOM_CRYSTAL_INDEX:.2f}")

heading("STATION BEAM RADII: BEFORE / AFTER")
for station, radius in stations:
    compare(station.label, initial.at(station).w, optimized.at(station).w, 1e3, "mm")
    print(f"  Target radius: {1e3 * radius:.3f} mm")
compare("Workpiece radius", initial.at(workpiece).w, optimized.at(workpiece).w, 1e6, "um")
compare("Workpiece waist offset", initial.at(workpiece).z0,
        optimized.at(workpiece).z0, 1e3, "mm")
print(f"Target workpiece waist: {1e6 * FOCUS_RADIUS:.1f} um "
      f"at z = {1e3 * WORKPIECE_Z:.0f} mm")

heading("MODEL CHECKS: BEFORE / AFTER")
compare("Worst divergence",
        max(state.divergence_angle for state in initial.states),
        max(state.divergence_angle for state in optimized.states), unit="rad")
print(f"Divergence limit: {MAX_DIVERGENCE:.3f} rad")
for element, diameter in apertures:
    compare(f"{element.label} radius", initial.at(element).w,
            optimized.at(element).w, 1e3, "mm")
    print(f"  Beam-radius limit: {1e3 * diameter / 4:.3f} mm")
compare("Lens 2–3 air clearance", z3.value - z2.value - lens2.d.value,
        solution[z3] - solution[z2] - solution[lens2.d], 1e3, "mm")

print("\n\n###### OBJECTIVES AT OPTIMUM ######")
for result in solution.targets:
    print(f"  {result.objective.label}: "
          f"residual={result.value:.3e}, cost={result.cost:.3e}")
