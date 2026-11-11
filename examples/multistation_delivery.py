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

import sys

from diffractix import (
    CompositeElement, 
    GaussianBeam, 
    Node, 
    Parameter, 
    Plane, 
    Slab, 
    Solver, 
    Space, 
    System,
    ThickLens, 
    ThinLens,
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



# Along the beam: expander, relay, expander, relay, expander, focusing lens.
# The expanders (negative) are ideal thin lenses.
expander1 = ThinLens(f=-160e-3, label="Expander 1").variable(f=(-400e-3, -45e-3))
expander2 = ThinLens(f=-340e-3, label="Expander 2").variable(f=(-500e-3, -45e-3))
expander3 = ThinLens(f=-160e-3, label="Expander 3").variable(f=(-500e-3, -45e-3))

# The positive lenses are biconvex thick lenses (R1 > 0, R2 < 0) in fixed glass.
relay1 = ThickLens(d=6e-3, n=GLASS_INDEX, R1=150e-3, R2=-250e-3, label="Relay 1")
relay1.variable(R1=(90e-3, 550e-3), R2=(-550e-3, -90e-3), d=(3e-3, 11e-3))
relay2 = ThickLens(d=8e-3, n=GLASS_INDEX, R1=240e-3, R2=-280e-3, label="Relay 2")
relay2.variable(R1=(90e-3, 500e-3), R2=(-500e-3, -90e-3), d=(3e-3, 12e-3))
focusing_lens = ThickLens(d=4e-3, n=GLASS_INDEX, R1=180e-3, R2=-220e-3, label="Focusing lens")
focusing_lens.variable(R1=(70e-3, 350e-3), R2=(-350e-3, -70e-3), d=(3e-3, 10e-3))

lenses = (expander1, relay1, expander2, relay2, expander3, focusing_lens)
thick_lenses = (relay1, relay2, focusing_lens)

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
aom.variable(d=(6e-3, 14e-3))
eom = CrystalModule(d=17e-3, n=EOM_CRYSTAL_INDEX, length=EOM_LENGTH, label="EOM")
eom.variable(d=(8e-3, 18e-3))

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

# Absolute position of each lens's front surface
expander1_z = Parameter(135e-3, name="Expander 1 position").variable()
relay1_z = Parameter(270e-3, name="Relay 1 position").variable()
expander2_z = Parameter(365e-3, name="Expander 2 position").variable()
relay2_z = Parameter(500e-3, name="Relay 2 position").variable()
expander3_z = Parameter(675e-3, name="Expander 3 position").variable()
focusing_z = Parameter(720e-3, name="Focusing lens position").variable()
positions = (expander1_z, relay1_z, expander2_z, relay2_z, expander3_z, focusing_z)

system = System(optical_input=source)
system.add(expander1, z=expander1_z)
system.add(aom, z=AOM_Z)
system.add(relay1, z=relay1_z)
system.add(expander2, z=expander2_z)
system.add(eom, z=EOM_Z)
system.add(relay2, z=relay2_z)
system.add(scanner, z=SCANNER_Z)
system.add(expander3, z=expander3_z)
system.add(focusing_lens, z=focusing_z)
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
solver.require(sim[expander2].z_start - sim[relay1.back].z_end >= MIN_LENS_GAP,
               label="Relay 1 to expander 2 gap")
solver.require(sim[workpiece].z_start - sim[focusing_lens.back].z_end >= MIN_WORKING_DISTANCE,
               label="Working distance")

# Paraxial limit: the Gaussian divergence stays below the limit after every element.
for stage in sim.stages:
    solver.require(stage.after.divergence_angle <= MAX_DIVERGENCE)

# Apertures. Between a module's entrance and exit the beam has no power, so
# checking both stations bounds it inside the module as well.
apertures = (
    (expander1, FIRST_LENS_DIAMETER),
    (expander2, THIN_LENS_DIAMETER),
    (expander3, THIN_LENS_DIAMETER),
    (relay1.front, THICK_LENS_DIAMETER),
    (relay1.back, THICK_LENS_DIAMETER),
    (relay2.front, THICK_LENS_DIAMETER),
    (relay2.back, THICK_LENS_DIAMETER),
    (focusing_lens.front, THICK_LENS_DIAMETER),
    (focusing_lens.back, THICK_LENS_DIAMETER),
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
# Soft goals count as reached within GOAL_TOLERANCE of their target (the waist
# offset within that fraction of the target Rayleigh range); hard requirements
# are reached when they hold (up to rounding).
GOAL_TOLERANCE = 0.01
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # the check marks need a UTF-8 console


def mark(reached):
    return "✓" if reached else "✗"


def near(value, target, tolerance):
    return abs(value - target) <= tolerance


def at_most(value, limit):
    return value <= limit * (1 + 1e-6)


def at_least(value, limit):
    return value >= limit * (1 - 1e-6)


def heading(title):
    print(f"\n{title}")
    print(f"  {'':34} {'Initial':>12}   {'Optimized':>12}   Goal")


def goal(label, before, after, reached, goal_text, scale=1.0):
    """One row: the initial and optimized value, each with its mark, then the goal."""
    before, after = float(before), float(after)
    print(f"  {label:<34} {mark(reached(before))} {before * scale:>10.3f}   "
          f"{mark(reached(after))} {after * scale:>10.3f}   {goal_text}")


def design(label, before, after, scale=1e3, unit="mm"):
    print(f"  {label:<34} {float(before) * scale:>12.2f}   {float(after) * scale:>12.2f}   {unit}")


def gap(result, front, back):
    return result[back].z_start - result[front].z_end


def worst_divergence(result):
    return max(float(state.divergence_angle) for state in result.states)


def worst_aperture_use(result):
    """Largest beam radius as a fraction of its limit (a quarter of the clear diameter)."""
    return max(float(result[element].after.w) / (diameter / 4) for element, diameter in apertures)


status = "converged" if solution.success else "did not converge"
feasibility = "all requirements hold" if solution.feasible else f"{len(solution.violations)} violated"
print(f"Optimization {status}, {feasibility} ({SOLVER_METHOD}, {solution.iterations} iterations, "
      f"{len(solution.x)} design variables)")
print(f"  {solution.message}")

heading("Design goals")
for station, radius in stations:
    goal(f"{station.label} beam radius", initial[station].after.w, optimized[station].after.w,
         lambda w, r=radius: near(w, r, GOAL_TOLERANCE * r), f"{radius * 1e3:.3f} mm ± 1 %", 1e3)
goal("Workpiece spot radius", initial[workpiece].after.w, optimized[workpiece].after.w,
     lambda w: near(w, FOCUS_RADIUS, GOAL_TOLERANCE * FOCUS_RADIUS),
     f"{FOCUS_RADIUS * 1e6:.0f} um ± 1 %", 1e6)
goal("Workpiece waist offset", initial[workpiece].after.z0, optimized[workpiece].after.z0,
     lambda z0: near(z0, 0.0, GOAL_TOLERANCE * target_focus.z_r),
     f"0 ± {GOAL_TOLERANCE * target_focus.z_r * 1e3:.2f} mm (1 % of the Rayleigh range)", 1e3)

heading("Requirements")
goal("Relay 1 to expander 2 air gap", gap(initial, relay1.back, expander2),
     gap(optimized, relay1.back, expander2),
     lambda d: at_least(d, MIN_LENS_GAP), f"≥ {MIN_LENS_GAP * 1e3:.0f} mm", 1e3)
goal("Working distance", gap(initial, focusing_lens.back, workpiece),
     gap(optimized, focusing_lens.back, workpiece),
     lambda d: at_least(d, MIN_WORKING_DISTANCE), f"≥ {MIN_WORKING_DISTANCE * 1e3:.0f} mm", 1e3)
goal("Worst divergence", worst_divergence(initial), worst_divergence(optimized),
     lambda angle: at_most(angle, MAX_DIVERGENCE), f"≤ {MAX_DIVERGENCE * 1e3:.0f} mrad", 1e3)
goal("Worst aperture use (w / limit)", worst_aperture_use(initial), worst_aperture_use(optimized),
     lambda use: at_most(use, 1.0), "≤ 1")

print("\nDesign")
print(f"  {'':34} {'Initial':>12}   {'Optimized':>12}")
for lens, position in zip(lenses, positions):
    design(f"{lens.label} position", position.value, solution.value(position))
    names = ("f",) if isinstance(lens, ThinLens) else ("R1", "R2", "d")
    for name in names:
        parameter = getattr(lens, name)
        design(f"  {name}", parameter.value, solution.value(parameter))
design("AOM crystal thickness", aom.d.value, solution.value(aom.d))
design("EOM crystal thickness", eom.d.value, solution.value(eom.d))
design("Total length", initial.z[-1], optimized.z[-1])


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
