"""Deliver a laser beam through fixed instrument stations to a focused workpiece spot.

The beam passes an acousto-optic modulator (AOM), an electro-optic modulator
(EOM) and a scanner, each at a fixed position, before being focused onto the
workpiece. The optimizer adjusts 20 design variables:

* six lens positions,
* three negative focal lengths,
* the radii and thicknesses of three positive thick lenses,
* the crystal thicknesses inside the AOM and EOM.

The modulator crystals are modelled only as passive glass slabs (refraction, no
modulation). All lengths are in metres; stations stay at fixed absolute positions.
"""

from time import perf_counter

from diffractix import (
    Constraint, GaussianBeam, Parameter, Plane, Slab, Solver, Space, System, ThickLens, ThinLens,
)

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
MAX_DIVERGENCE = 0.03  # Gaussian far-field half-angle in radians: paraxial limit

THIN_LENS_DIAMETER = 9e-3
FIRST_LENS_DIAMETER = 5e-3
THICK_LENS_DIAMETER = 9e-3
STATION_DIAMETERS = (6e-3, 6e-3, 8e-3, 8e-3, 8e-3)

# SLSQP suits this heavily constrained design; trust-constr uses other options.
SOLVER_METHOD = "SLSQP"  # Alternative: "trust-constr"
SOLVER_OPTIONS = {
    "SLSQP": {"maxiter": 650, "ftol": 1e-11},
    "trust-constr": {"maxiter": 350, "gtol": 1e-7, "xtol": 1e-9},
}


# -----------------
# SOURCE AND TARGET
# -----------------
source = GaussianBeam.from_waist(w0=INPUT_RADIUS, wavelength=WAVELENGTH)
target_focus = GaussianBeam.from_waist(w0=FOCUS_RADIUS, wavelength=WAVELENGTH).numeric()
if source.numeric().divergence_angle > MAX_DIVERGENCE:
    raise ValueError("The source exceeds the paraxial divergence limit.")


# ------
# LENSES
# ------
# The starting design is deliberately poor but feasible: the workpiece spot is
# about 124 um instead of 45 um, with its waist about 18 mm off target.

# Absolute position of each lens's front surface. Disjoint bounds keep the
# lenses in order around the fixed stations.
z1 = Parameter(135e-3, name="Lens 1 position").variable(bounds=(40e-3, 170e-3))
z2 = Parameter(270e-3, name="Lens 2 position").variable(bounds=(240e-3, 330e-3))
z3 = Parameter(365e-3, name="Lens 3 position").variable(bounds=(345e-3, 400e-3))
z4 = Parameter(500e-3, name="Lens 4 position").variable(bounds=(475e-3, 585e-3))
z5 = Parameter(675e-3, name="Lens 5 position").variable(bounds=(655e-3, 705e-3))
z6 = Parameter(720e-3, name="Lens 6 position").variable(bounds=(715e-3, 730e-3))
positions = (z1, z2, z3, z4, z5, z6)

# Negative lenses are ideal thin lenses.
lens1 = ThinLens(f=-160e-3, label="Lens 1 (expander)")
lens1.f.variable(bounds=(-400e-3, -45e-3))
lens3 = ThinLens(f=-340e-3, label="Lens 3 (expander)")
lens3.f.variable(bounds=(-500e-3, -45e-3))
lens5 = ThinLens(f=-160e-3, label="Lens 5 (expander)")
lens5.f.variable(bounds=(-500e-3, -45e-3))

# Positive lenses are biconvex thick lenses (R1 > 0, R2 < 0) in fixed glass.
lens2 = ThickLens(d=6e-3, n=GLASS_INDEX, R1=150e-3, R2=-250e-3, label="Lens 2 (thick relay)")
lens2.R1.variable(bounds=(90e-3, 550e-3))
lens2.R2.variable(bounds=(-550e-3, -90e-3))
lens2.d.variable(bounds=(3e-3, 11e-3))

lens4 = ThickLens(d=8e-3, n=GLASS_INDEX, R1=240e-3, R2=-280e-3, label="Lens 4 (thick relay)")
lens4.R1.variable(bounds=(90e-3, 500e-3))
lens4.R2.variable(bounds=(-500e-3, -90e-3))
lens4.d.variable(bounds=(3e-3, 12e-3))

lens6 = ThickLens(d=4e-3, n=GLASS_INDEX, R1=180e-3, R2=-220e-3, label="Lens 6 (thick focusing)")
lens6.R1.variable(bounds=(70e-3, 350e-3))
lens6.R2.variable(bounds=(-350e-3, -70e-3))
lens6.d.variable(bounds=(3e-3, 10e-3))

lenses = (lens1, lens2, lens3, lens4, lens5, lens6)
thin_lenses = (lens1, lens3, lens5)
thick_lenses = (lens2, lens4, lens6)

# Name the curved surfaces of each thick lens for the constraint labels and tables.
for index, lens in zip((2, 4, 6), thick_lenses):
    lens.front.label = f"Lens {index} front surface"
    lens.back.label = f"Lens {index} back surface"


# --------
# STATIONS
# --------
# The AOM and EOM modules have fixed lengths. A thicker crystal shrinks the air
# allowances on either side; the module entrance and exit stay where they are.
aom_crystal = Slab(d=9e-3, n=AOM_CRYSTAL_INDEX, label="AOM crystal")
aom_crystal.d.variable(bounds=(6e-3, 14e-3))
eom_crystal = Slab(d=17e-3, n=EOM_CRYSTAL_INDEX, label="EOM crystal")
eom_crystal.d.variable(bounds=(8e-3, 18e-3))

aom, aom_exit = Plane(label="AOM"), Plane(label="AOM exit")
eom, eom_exit = Plane(label="EOM"), Plane(label="EOM exit")
scanner, workpiece = Plane(label="Scanner"), Plane(label="Workpiece")
stations = (
    (aom, AOM_RADIUS),
    (aom_exit, AOM_EXIT_RADIUS),
    (eom, EOM_RADIUS),
    (eom_exit, EOM_EXIT_RADIUS),
    (scanner, SCANNER_RADIUS),
)


def allowance(module_length, crystal):
    """Air on one side of a crystal centred in a fixed-length module."""
    return (module_length - crystal.d) / 2


# ------
# SYSTEM
# ------
# Gaps are written from the absolute coordinates. A thick lens occupies its
# thickness after its front surface, so it is subtracted from the next gap.
working_distance = Space(d=WORKPIECE_Z - z6 - lens6.d, label="Working distance")
lens2_to_lens3 = Space(d=z3 - z2 - lens2.d, label="Lens 2 to lens 3")
system = System(
    Space(d=z1, label="Input gap"),
    lens1,
    Space(d=AOM_Z - z1, label="To AOM"),
    aom,
    Space(d=allowance(AOM_EXIT_Z - AOM_Z, aom_crystal), label="AOM input allowance"),
    aom_crystal,
    Space(d=allowance(AOM_EXIT_Z - AOM_Z, aom_crystal), label="AOM output allowance"),
    aom_exit,
    Space(d=z2 - AOM_EXIT_Z, label="To lens 2"),
    lens2,
    lens2_to_lens3,
    lens3,
    Space(d=EOM_Z - z3, label="To EOM"),
    eom,
    Space(d=allowance(EOM_EXIT_Z - EOM_Z, eom_crystal), label="EOM input allowance"),
    eom_crystal,
    Space(d=allowance(EOM_EXIT_Z - EOM_Z, eom_crystal), label="EOM output allowance"),
    eom_exit,
    Space(d=z4 - EOM_EXIT_Z, label="To lens 4"),
    lens4,
    Space(d=SCANNER_Z - z4 - lens4.d, label="To scanner"),
    scanner,
    Space(d=z5 - SCANNER_Z, label="To lens 5"),
    lens5,
    Space(d=z6 - z5, label="Lens 5 to lens 6"),
    lens6,
    working_distance,
    workpiece,
    optical_input=source,
)
system.require(
    working_distance.d >= MIN_WORKING_DISTANCE,
    lens2_to_lens3.d >= MIN_LENS_GAP,
)

simulation = system.build()
initial = simulation.run()


# ------------
# OPTIMIZATION
# ------------
solver = Solver(simulation)

# Beam radius at every station, then a focused spot with its waist on the workpiece.
for station, radius in stations:
    solver.target(
        value=lambda sim, station=station: sim[station].after.w,
        target=radius,
        scale=radius,
        label=f"{station.label} beam radius",
    )
solver.target(
    value=lambda sim: sim[workpiece].after.w,
    target=FOCUS_RADIUS,
    scale=FOCUS_RADIUS,
    label="Workpiece spot radius",
)
solver.target(
    value=lambda sim: sim[workpiece].after.z0,
    target=0.0,
    scale=target_focus.z_r,
    label="Workpiece waist position",
)

# Soft manufacturing preferences remove otherwise arbitrary freedom: symmetric
# biconvex lenses, about 6 mm centre thickness, familiar crystal lengths.
for index, lens in zip((2, 4, 6), thick_lenses):
    solver.target((lens.R1 + lens.R2) / 0.10, label=f"Lens {index} surface symmetry")
    solver.target((lens.d - 6e-3) / 12e-3, label=f"Lens {index} preferred thickness")
for label, crystal, preferred in (("AOM", aom_crystal, 10e-3), ("EOM", eom_crystal, 14e-3)):
    solver.target((crystal.d - preferred) / 20e-3, label=f"{label} preferred crystal thickness")

# Only surfaces with optical power change the divergence; the flat slab faces do
# not. For a thick lens, both curved surfaces are checked.
power_elements = (
    *thin_lenses,
    *(surface for lens in thick_lenses for surface in (lens.front, lens.back)),
)
for element in power_elements:
    solver.require(Constraint(
        lambda sim, element=element: sim[element].after.divergence_angle,
        upper_bound=MAX_DIVERGENCE,
        label=f"Paraxial divergence after {element.label}",
    ))

# Keep the beam radius below a quarter of each clear diameter (two beam radii of
# clearance). Between a slab module's entrance and exit the beam has no power,
# so checking both stations bounds it inside the module as well.
apertures = (
    (lens1, FIRST_LENS_DIAMETER),
    (lens3, THIN_LENS_DIAMETER),
    (lens5, THIN_LENS_DIAMETER),
    *((surface, THICK_LENS_DIAMETER) for lens in thick_lenses for surface in (lens.front, lens.back)),
    *zip((station for station, _ in stations), STATION_DIAMETERS),
)
for element, diameter in apertures:
    solver.require(Constraint(
        lambda sim, element=element: sim[element].after.w,
        upper_bound=diameter / 4,
        label=f"{element.label} aperture clearance",
    ))

start = perf_counter()
solution = solver.solve(method=SOLVER_METHOD, options=SOLVER_OPTIONS[SOLVER_METHOD])
elapsed = perf_counter() - start
optimized = solution.run()


# -------
# RESULTS
# -------
def table(title):
    print(f"\n{title}")
    print(f"{'':32} {'Initial':>12} {'Optimized':>12}")
    print("-" * 62)


def row(label, before, after, scale=1.0, unit=""):
    print(f"{label:<32} {before * scale:>12.4f} {after * scale:>12.4f} {unit}")


print(f"Optimization ({SOLVER_METHOD}): success={solution.success}, "
      f"feasible={solution.feasible}, cost={solution.cost:.3e}, "
      f"iterations={solution.iterations}, evaluations={solution.function_evaluations}, "
      f"time={elapsed:.2f} s")
print(f"  {solution.message}")
print(f"  Design variables: {len(solution.x)}")

table("Design")
for index, (position, lens) in enumerate(zip(positions, lenses), start=1):
    row(f"Lens {index} position", position.value, solution.value(position), 1e3, "mm")
    names = ("f",) if isinstance(lens, ThinLens) else ("R1", "R2", "d")
    for name in names:
        parameter = getattr(lens, name)
        row(f"Lens {index} {name}", parameter.value, solution.value(parameter), 1e3, "mm")
row("AOM crystal thickness", aom_crystal.d.value, solution.value(aom_crystal.d), 1e3, "mm")
row("EOM crystal thickness", eom_crystal.d.value, solution.value(eom_crystal.d), 1e3, "mm")
row("Lens 2 to lens 3 air gap", lens2_to_lens3.d.value, solution.value(lens2_to_lens3.d), 1e3, "mm")
row("Working distance", working_distance.d.value, solution.value(working_distance.d), 1e3, "mm")
row("Total length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Minimums: lens gap {MIN_LENS_GAP * 1e3:.0f} mm, working distance {MIN_WORKING_DISTANCE * 1e3:.0f} mm")

table("Beam at the stations")
for station, radius in stations:
    row(f"{station.label} radius", initial[station].after.w, optimized[station].after.w, 1e3, "mm")
    print(f"  target {radius * 1e3:.3f} mm")
row("Workpiece spot radius", initial[workpiece].after.w, optimized[workpiece].after.w, 1e6, "um")
row("Workpiece waist offset", initial[workpiece].after.z0, optimized[workpiece].after.z0, 1e3, "mm")
print(f"Target: {FOCUS_RADIUS * 1e6:.0f} um waist at z = {WORKPIECE_Z * 1e3:.0f} mm")

table("Model checks")
for element in power_elements:
    row(f"{element.label} divergence", initial[element].after.divergence_angle,
        optimized[element].after.divergence_angle, unit="rad")
print(f"Divergence limit: {MAX_DIVERGENCE} rad")
for element, diameter in apertures:
    row(f"{element.label} radius", initial[element].after.w, optimized[element].after.w, 1e3, "mm")
    print(f"  limit {diameter / 4 * 1e3:.3f} mm")

print("\nObjectives at the optimum")
for objective in solution.objectives:
    print(f"  {objective.label}: residual={objective.value:.3e}, cost={objective.cost:.3e}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    figure.suptitle("Multi-station delivery")
    initial.plot(ax=axes[0])
    optimized.plot(ax=axes[1])
    axes[0].set_title("Before optimization", loc="left")
    axes[1].set_title("After optimization", loc="left")
    axes[0].set_xlabel("")
    plt.show()
