"""Constrained Gaussian fiber-to-fiber mode matching with packaged thin lenses.

Matplotlib is included with Diffractix for the before/after plots.
"""

from time import perf_counter

from diffractix import (
    CompositeElement,
    Constraint,
    GaussianBeam,
    Node,
    Plane,
    Solver,
    Space,
    System,
    ThinLens,
)

# All lengths are in meters.
WAVELENGTH = 1550e-9
LENS_LENGTH = 4e-3
MAX_PACKAGE_LENGTH = 30e-3
DESIRED_PACKAGE_LENGTH = 25e-3

# Engineering limits for the ideal Gaussian / paraxial model.
MAX_DIVERGENCE = 0.15       # Gaussian far-field half-angle, radians
LENS_CLEAR_DIAMETER = 4e-3  # Illustrative clear aperture


class PackagedThinLens(CompositeElement):
    """Ideal thin lens centered in a fixed-length free-space allowance."""

    f: Node

    def __init__(self, f: float, length: float, label: str):
        self.f = f
        self.label = label
        self.front = Space(d=length / 2, label=f"{label} front")
        self.lens = ThinLens(f=self.f, label=label)
        self.back = Space(d=length / 2, label=f"{label} back")
        super().__init__()

    @property
    def element_length(self):
        return sum(element.element_length for element in self.leaf_elements)




#------
# SETUP
#------
# Input and target fiber modes
source = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=10.4e-6)
target_mode = GaussianBeam.from_fiber_tip(wavelength=WAVELENGTH, MFD=9.2e-6).numeric()

if max(source.numeric().divergence_angle, target_mode.divergence_angle) > MAX_DIVERGENCE:
    raise ValueError("Source or target mode exceeds the paraxial divergence limit.")

# Intentionally imperfect starting design (~7% overlap in the ideal ABCD model).
# It still respects the design-variable bounds and physical hard constraints.
input_gap = Space(d=5e-3, label="Input gap")
input_gap.d.variable()

lens1 = PackagedThinLens(f=5e-3, length=LENS_LENGTH, label="Lens 1")
lens1.f.variable(bounds=(4e-3, 15e-3))

middle_gap = Space(d=5e-3, label="Lens spacing")
middle_gap.d.variable()

lens2 = PackagedThinLens(f=5e-3, length=LENS_LENGTH, label="Lens 2")
lens2.f.variable(bounds=(4e-3, 15e-3))

output_gap = Space(d=5e-3, label="Output gap")
output_gap.d.variable()

target = Plane(label="Target fiber")

system = System(
    input_gap, 
    lens1, 
    middle_gap, 
    lens2, 
    output_gap, 
    target,
    input=source,
)

# maximum system length: lens packages contribute their own 3 mm lengths.
package_length = sum(element.element_length for element in system.elements)
system.require(package_length <= MAX_PACKAGE_LENGTH)


#-----------
# SIMULATION
#-----------
simulation = system.build()

# Check paraxiality after every compiled optical element.
# states[0] is the source; states[i] follows element_info[i - 1].
for i, info in enumerate(simulation.element_info, start=1):
    simulation.require(
        Constraint(
            lambda ctx, i=i: ctx.states[i].divergence_angle,
            upper_bound=MAX_DIVERGENCE,
            label=f"Paraxial divergence after {info.label or info.path or info.type_name}",
        )
    )

# Require at least two beam radii of clearance at each lens.
for lens in (lens1, lens2):
    simulation.require(
        Constraint(
            lambda ctx, lens=lens: ctx.at(lens.lens).w,
            upper_bound=LENS_CLEAR_DIAMETER / 4,
            label=f"{lens.label} aperture clearance",
        )
    )



#-------
# SOLVER
#-------
# Optical targets
solver = Solver(simulation)
solver.target(
    value=lambda ctx: ctx.at(target).w,
    target=target_mode.w,
    scale=target_mode.w,
    label="Target mode radius",
)
solver.target(
    value=lambda ctx: ctx.at(target).z0,
    target=0.0,
    scale=target_mode.z_r,
    label="Target waist position",
)
# Soft preference: package <=25 mm, with a hard 30 mm system limit.
solver.target(
    ((package_length - DESIRED_PACKAGE_LENGTH) / 1e-3).maximum(0.0),
    label="Package length over 25 mm",
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
print(f"Success:             {solution.success}")
print(f"Feasible:            {solution.feasible}")
print(f"Termination:         {solution.message}")
print(f"Final objective:     {solution.cost:.3e}")
print(f"Solve time:          {elapsed:.3f} s")

# The current Solution API does not yet expose iteration diagnostics publicly.
# This is the only private access in the example, limited to reporting.
backend_result = solution._result
print(f"Iterations:          {backend_result.iterations}")
print(f"Function evaluations: {backend_result.raw.nfev}")

print("\n\n###### DESIGN: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
for label, parameter in (
    ("Input gap", input_gap.d),
    ("Lens 1 focal length", lens1.f),
    ("Lens spacing", middle_gap.d),
    ("Lens 2 focal length", lens2.f),
    ("Output gap", output_gap.d),
):
    compare(label, parameter.value, solution[parameter], 1e3, "mm")
compare("Package length", initial.z[-1], optimized.z[-1], 1e3, "mm")
print(f"Lens packages: {1e3 * LENS_LENGTH:.1f} mm each")
print(f"Preferred length: {1e3 * DESIRED_PACKAGE_LENGTH:.1f} mm; "
      f"maximum: {1e3 * MAX_PACKAGE_LENGTH:.1f} mm")

initial_beam = initial.at(target)
matched_beam = optimized.at(target)
initial_overlap = initial_beam.overlap_with(target_mode)
matched_overlap = matched_beam.overlap_with(target_mode)

print("\n\n###### OPTICAL PERFORMANCE: BEFORE / AFTER ###### ")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare("Beam radius", initial_beam.w, matched_beam.w, 1e6, "um")
compare("Waist offset", initial_beam.z0, matched_beam.z0, 1e6, "um")
compare("Fiber overlap", initial_overlap, matched_overlap, 100, "%")
print(f"Target mode radius: {1e6 * target_mode.w:.4f} um")

print("\n\n###### MODEL CHECKS: BEFORE / AFTER ######")
print(f"{'':23} {'Initial':>12} {'Optimized':>12}")
print("-" * 52)
compare(
    "Worst divergence",
    max(state.divergence_angle for state in initial.states),
    max(state.divergence_angle for state in optimized.states),
    unit="rad",
)
for lens in (lens1, lens2):
    compare(
        f"{lens.label} beam radius",
        initial.at(lens.lens).w,
        optimized.at(lens.lens).w,
        1e3,
        "mm",
    )
print(f"Divergence limit: {MAX_DIVERGENCE:.3f} rad; "
      f"beam-radius limit: {1e3 * LENS_CLEAR_DIAMETER / 4:.3f} mm")

print("\n\n###### OBJECTIVES AT OPTIMUM ###### ")
for result in solution.targets:
    print(f"  {result.objective.label}: "
          f"residual={result.value:.3e}, cost={result.cost:.3e}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Fiber mode matching", fontsize=12)
    axes[0].set_title("Before optimization", loc="left", fontsize=11)
    axes[1].set_title("After optimization", loc="left", fontsize=11)
    initial.plot(ax=axes[0])
    optimized.plot(ax=axes[1])
    axes[0].set_xlabel("")
    plt.show()
