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

from diffractix import (
    CompositeElement, GaussianBeam, Node, Parameter, Plane, Slab, Solver, Space, System,
    ThickLens, ThinLens,
)

WAVELENGTH = 1064e-9
INPUT_RADIUS = 0.65e-3

AOM_Z, AOM_LENGTH = 200e-3, 15e-3
EOM_Z, EOM_LENGTH = 430e-3, 20e-3
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

# Clear diameters. The beam radius must stay below a quarter of each, which keeps
# two beam radii of clearance (clipping itself is not modelled).
FIRST_LENS_DIAMETER = 5e-3
THIN_LENS_DIAMETER = 9e-3
THICK_LENS_DIAMETER = 9e-3
AOM_DIAMETER = 6e-3
EOM_DIAMETER = 8e-3
SCANNER_DIAMETER = 8e-3

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
z1 = Parameter(135e-3, name="Lens 1 position").variable()
z2 = Parameter(270e-3, name="Lens 2 position").variable()
z3 = Parameter(365e-3, name="Lens 3 position").variable()
z4 = Parameter(500e-3, name="Lens 4 position").variable()
z5 = Parameter(675e-3, name="Lens 5 position").variable()
z6 = Parameter(720e-3, name="Lens 6 position").variable()
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
thick_lenses = (lens2, lens4, lens6)

# Name the curved surfaces of each thick lens for the constraint labels.
for lens in thick_lenses:
    lens.front.label = f"{lens.label} front surface"
    lens.back.label = f"{lens.label} back surface"


# --------
# STATIONS
# --------
class CrystalModule(CompositeElement):
    """A passive crystal centred in a module of fixed length, between two station planes."""

    d: Node  # crystal thickness

    def __init__(self, d, n, length, label):
        self.d = d
        self.label = label
        self.entrance = Plane(label=f"{label} entrance")
        self.air_before = Space(d=(length - self.d) / 2)
        self.crystal = Slab(d=self.d, n=n, label=f"{label} crystal")
        self.air_after = Space(d=(length - self.d) / 2)
        self.exit = Plane(label=f"{label} exit")
        super().__init__()


aom = CrystalModule(d=9e-3, n=AOM_CRYSTAL_INDEX, length=AOM_LENGTH, label="AOM")
aom.d.variable(bounds=(6e-3, 14e-3))
eom = CrystalModule(d=17e-3, n=EOM_CRYSTAL_INDEX, length=EOM_LENGTH, label="EOM")
eom.d.variable(bounds=(8e-3, 18e-3))

scanner, workpiece = Plane(label="Scanner"), Plane(label="Workpiece")
stations = (
    (aom.entrance, AOM_RADIUS),
    (aom.exit, AOM_EXIT_RADIUS),
    (eom.entrance, EOM_RADIUS),
    (eom.exit, EOM_EXIT_RADIUS),
    (scanner, SCANNER_RADIUS),
)


# ------
# SYSTEM
# ------
# Every element is placed at an absolute position (its front surface). The air gaps
# between them are generated automatically, and a gap that would be negative is
# infeasible, which keeps the lenses in order.
system = System(optical_input=source)
system.add(lens1, z=z1)
system.add(aom, z=AOM_Z)
system.add(lens2, z=z2)
system.add(lens3, z=z3)
system.add(eom, z=EOM_Z)
system.add(lens4, z=z4)
system.add(scanner, z=SCANNER_Z)
system.add(lens5, z=z5)
system.add(lens6, z=z6)
system.add(workpiece, z=WORKPIECE_Z)

simulation = system.build()
initial = simulation.run()


# ------------
# OPTIMIZATION
# ------------
solver = Solver(simulation)
sim = simulation.symbolic  # the result as expressions, for objectives and constraints

# Beam radius at every station, then a focused spot with its waist on the workpiece.
# The default scale of a target is its own size, so these are relative errors.
for station, radius in stations:
    solver.target(value=sim[station].after.w, target=radius, label=f"{station.label} beam radius")
solver.target(value=sim[workpiece].after.w, target=FOCUS_RADIUS, label="Workpiece spot radius")
solver.target(value=sim[workpiece].after.z0, target=0.0, scale=target_focus.z_r,
              label="Workpiece waist position")

# Soft manufacturing preferences remove otherwise arbitrary freedom: 
# symmetric biconvex lenses, about 6 mm centre thickness, familiar crystal lengths.
for lens in thick_lenses:
    solver.target(value=lens.R1 + lens.R2, target=0, scale=0.10, label=f"{lens.label} surface symmetry")
    solver.target(value=lens.d, target=6e-3, scale=12e-3, label=f"{lens.label} preferred thickness")
solver.target(value=aom.d, target=10e-3, scale=20e-3, label="AOM preferred crystal thickness")
solver.target(value=eom.d, target=14e-3, scale=20e-3, label="EOM preferred crystal thickness")


# Air gaps, measured on the result so they follow every thickness and position.
solver.require(sim[lens3].z_start - sim[lens2.back].z_end >= MIN_LENS_GAP, label="Lens 2 to lens 3 gap")
solver.require(sim[workpiece].z_start - sim[lens6.back].z_end >= MIN_WORKING_DISTANCE, label="Working distance")

# Paraxial limit: the Gaussian divergence stays below the limit after every element.
for stage in sim.stages:
    solver.require(stage.after.divergence_angle <= MAX_DIVERGENCE)

# Apertures. Between a module's entrance and exit the beam has no power, so
# checking both stations bounds it inside the module as well.
apertures = (
    (lens1, FIRST_LENS_DIAMETER),
    (lens3, THIN_LENS_DIAMETER),
    (lens5, THIN_LENS_DIAMETER),
    (lens2.front, THICK_LENS_DIAMETER),
    (lens2.back, THICK_LENS_DIAMETER),
    (lens4.front, THICK_LENS_DIAMETER),
    (lens4.back, THICK_LENS_DIAMETER),
    (lens6.front, THICK_LENS_DIAMETER),
    (lens6.back, THICK_LENS_DIAMETER),
    (aom.entrance, AOM_DIAMETER),
    (aom.exit, AOM_DIAMETER),
    (eom.entrance, EOM_DIAMETER),
    (eom.exit, EOM_DIAMETER),
    (scanner, SCANNER_DIAMETER),
)
for element, diameter in apertures:
    solver.require(sim[element].after.w <= diameter / 4, label=f"{element.label} aperture")

solution = solver.solve(method=SOLVER_METHOD, options=SOLVER_OPTIONS[SOLVER_METHOD])
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
      f"iterations={solution.iterations}, evaluations={solution.function_evaluations}")
print(f"  {solution.message}")
print(f"  Design variables: {len(solution.x)}")

table("Design")
for index, (position, lens) in enumerate(zip(positions, lenses), start=1):
    row(f"Lens {index} position", position.value, solution.value(position), 1e3, "mm")
    names = ("f",) if isinstance(lens, ThinLens) else ("R1", "R2", "d")
    for name in names:
        parameter = getattr(lens, name)
        row(f"Lens {index} {name}", parameter.value, solution.value(parameter), 1e3, "mm")
row("AOM crystal thickness", aom.d.value, solution.value(aom.d), 1e3, "mm")
row("EOM crystal thickness", eom.d.value, solution.value(eom.d), 1e3, "mm")
row("Lens 2 to lens 3 air gap", initial[lens3].z_start - initial[lens2.back].z_end,
    optimized[lens3].z_start - optimized[lens2.back].z_end, 1e3, "mm")
row("Working distance", initial[workpiece].z_start - initial[lens6.back].z_end,
    optimized[workpiece].z_start - optimized[lens6.back].z_end, 1e3, "mm")
row("Total length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Minimums: lens gap {MIN_LENS_GAP * 1e3:.0f} mm, working distance {MIN_WORKING_DISTANCE * 1e3:.0f} mm")

table("Beam at the stations")
for station, radius in stations:
    row(f"{station.label} radius", initial[station].after.w, optimized[station].after.w, 1e3, "mm")
    print(f"  target {radius * 1e3:.3f} mm")
row("Workpiece spot radius", initial[workpiece].after.w, optimized[workpiece].after.w, 1e6, "um")
row("Workpiece waist offset", initial[workpiece].after.z0, optimized[workpiece].after.z0, 1e3, "mm")
print(f"Target: {FOCUS_RADIUS * 1e6:.0f} um waist at z = {WORKPIECE_Z * 1e3:.0f} mm")
row("Worst divergence", max(state.divergence_angle for state in initial.states),
    max(state.divergence_angle for state in optimized.states), unit="rad")
print(f"Divergence limit: {MAX_DIVERGENCE} rad")

print("\nConstraints at the optimum (SI units)")
for constraint in solution.constraints:
    if constraint.label:
        limit = constraint.upper_bound if constraint.upper_bound != float("inf") else constraint.lower_bound
        print(f"  {constraint.label:<46} value={float(constraint.value):.4g}  "
              f"limit={float(limit):.4g}  margin={float(constraint.margin):.3g}")

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
