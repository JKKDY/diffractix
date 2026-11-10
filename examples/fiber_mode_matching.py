"""Couple light from one single-mode fiber into another with two packaged lenses.

The fibers have different mode field diameters. The optimizer chooses three air
gaps and two focal lengths so that the beam arriving at the second fiber matches
its mode, while the package stays short. All lengths are in metres.
"""

from diffractix import (
    CompositeElement, GaussianBeam, Node, Plane, Solver, Space, System, ThinLens,
)

WAVELENGTH = 1550e-9
LENS_LENGTH = 4e-3
MAX_PACKAGE_LENGTH = 30e-3
PREFERRED_PACKAGE_LENGTH = 25e-3
MAX_DIVERGENCE = 0.15       # Gaussian far-field half-angle in radians: paraxial limit
LENS_CLEAR_DIAMETER = 4e-3  # Illustrative clear aperture


class PackagedThinLens(CompositeElement):
    """An ideal thin lens centred in a housing of fixed length."""

    f: Node

    def __init__(self, f, length, label):
        self.f = f
        self.label = label
        self.front = Space(d=length / 2, label=f"{label} front")
        self.lens = ThinLens(f=self.f, label=label)
        self.back = Space(d=length / 2, label=f"{label} back")
        super().__init__()


# ------
# DESIGN
# ------
source = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=10.4e-6)
target_mode = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=9.2e-6).numeric()
if max(source.numeric().divergence_angle, target_mode.divergence_angle) > MAX_DIVERGENCE:
    raise ValueError("A fiber mode exceeds the paraxial divergence limit.")

# A poor starting design (about 7% overlap) that still respects every limit.
input_gap = Space(d=5e-3, label="Input gap")
lens1 = PackagedThinLens(f=5e-3, length=LENS_LENGTH, label="Lens 1")
middle_gap = Space(d=5e-3, label="Lens spacing")
lens2 = PackagedThinLens(f=5e-3, length=LENS_LENGTH, label="Lens 2")
output_gap = Space(d=5e-3, label="Output gap")
target = Plane(label="Target fiber")

for gap in (input_gap, middle_gap, output_gap):
    gap.d.variable()  # Space already requires d >= 0.
for package in (lens1, lens2):
    package.f.variable(bounds=(4e-3, 15e-3))

system = System(input_gap, lens1, middle_gap, lens2, output_gap, target, optical_input=source)
package_length = sum(element.element_length for element in system.elements)
system.require(package_length <= MAX_PACKAGE_LENGTH)

simulation = system.build()
initial = simulation.run()


# ------------
# OPTIMIZATION
# ------------
solver = Solver(simulation)
sim = simulation.symbolic  # the result as expressions, for objectives and constraints

# Match the target fiber's mode: same radius (a relative error, since the default scale
# of a target is its own size), and the waist on the fiber face within one Rayleigh range.
solver.target(value=sim[target].after.w, target=target_mode.w, label="Mode radius")
solver.target(value=sim[target].after.z0, target=0.0, scale=target_mode.z_r, label="Waist position")
# A soft preference: penalize only the length beyond 25 mm (hard limit 30 mm).
solver.target(
    ((package_length - PREFERRED_PACKAGE_LENGTH) / 1e-3).maximum(0.0),
    label="Package length over 25 mm",
)

# Only the lenses change the divergence, so checking after each lens covers the
# whole path. Each lens also keeps two beam radii of aperture clearance.
for package in (lens1, lens2):
    solver.require(sim[package.lens].after.divergence_angle <= MAX_DIVERGENCE,
                   label=f"Paraxial divergence after {package.label}")
    solver.require(sim[package.lens].after.w <= LENS_CLEAR_DIAMETER / 4,
                   label=f"{package.label} aperture clearance")

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
for label, parameter in (
    ("Input gap", input_gap.d),
    ("Lens 1 focal length", lens1.f),
    ("Lens spacing", middle_gap.d),
    ("Lens 2 focal length", lens2.f),
    ("Output gap", output_gap.d),
):
    row(label, parameter.value, solution.value(parameter), 1e3, "mm")
row("Package length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Preferred {PREFERRED_PACKAGE_LENGTH * 1e3:.0f} mm, maximum {MAX_PACKAGE_LENGTH * 1e3:.0f} mm")

before, after = initial[target].after, optimized[target].after
table("Coupling into the target fiber")
row("Beam radius", before.w, after.w, 1e6, "um")
row("Waist offset", before.z0, after.z0, 1e6, "um")
row("Mode overlap", before.overlap_with(target_mode), after.overlap_with(target_mode), 100, "%")
print(f"Target mode radius: {target_mode.w * 1e6:.2f} um")

table("Model checks")
for package in (lens1, lens2):
    row(f"{package.label} beam radius", initial[package.lens].after.w,
        optimized[package.lens].after.w, 1e3, "mm")
    row(f"{package.label} divergence", initial[package.lens].after.divergence_angle,
        optimized[package.lens].after.divergence_angle, unit="rad")
print(f"Limits: radius {LENS_CLEAR_DIAMETER / 4 * 1e3:.2f} mm, divergence {MAX_DIVERGENCE} rad")

print("\nObjectives at the optimum")
for objective in solution.objectives:
    print(f"  {objective.label}: residual={objective.value:.3e}, cost={objective.cost:.3e}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    figure.suptitle("Fiber mode matching")
    initial.plot(ax=axes[0])
    optimized.plot(ax=axes[1])
    axes[0].set_title("Before optimization", loc="left")
    axes[1].set_title("After optimization", loc="left")
    axes[0].set_xlabel("")
    plt.show()
