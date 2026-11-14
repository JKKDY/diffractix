"""Complete design workflows through the public API only.

Each test declares a system with unit constants, builds it, optimizes it and
checks the result against physics worked out independently of the library
(geometric optics by hand, or plain NumPy ray-transfer matrices). They cover
what a user does from the first line to the report and the plot.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from diffractix import (  # noqa: E402
    CompositeElement, GaussianBeam, Node, Parameter, Plane, RayBundle, Solver, Space, System,
    ThickLens, ThinLens, units,
)
from diffractix.units import mm, mrad, nm, um  # noqa: E402

WAVELENGTH = 1064 * nm


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ----------------
# GAUSSIAN DESIGNS
# ----------------
def test_galilean_expander_design_reaches_the_geometric_solution_and_reports_it():
    """A 4x expander: f2 = -4 f1 and a spacing of f1 + f2, as geometric optics predicts."""
    source = GaussianBeam.from_waist(w0=0.5 * mm, wavelength=WAVELENGTH)
    diverging = ThinLens(f=-20 * mm, label="Diverging lens")
    spacing = Space(d=80 * mm, label="Lens spacing").variable(d=(40 * mm, 90 * mm))
    collimating = ThinLens(f=100 * mm, label="Collimating lens").variable(f=(60 * mm, 125 * mm))
    output = Plane(label="Output")
    system = System(diverging, spacing, collimating, Space(d=10 * mm), output, optical_input=source)
    system.require(spacing.d + 10 * mm <= 100 * mm, label="Total length")

    simulation = system.build()
    initial = simulation.run()
    sim = simulation.symbolic
    solver = Solver(simulation)
    solver.target(value=sim[output].after.w, target=4 * initial.source.w, label="Output radius")
    solver.target(value=sim[output].after.divergence_angle, target=initial.source.divergence_angle / 4,
                  label="Output divergence")
    solver.require(sim[collimating].after.w <= 2.5 * mm, label="Collimating lens aperture")
    solution = solver.solve(options={"ftol": 1e-14})
    optimized = solution.run()

    assert solution.success and solution.feasible
    assert solution.value(collimating.f) == pytest.approx(80 * mm, rel=2e-3)
    assert solution.value(spacing.d) == pytest.approx(60 * mm, rel=2e-3)
    assert float(optimized[output].after.w) == pytest.approx(2.0 * mm, rel=1e-3)
    assert float(optimized[output].after.divergence_angle) == pytest.approx(
        float(initial.source.divergence_angle) / 4, rel=1e-3)

    # The declaration and the built simulation still describe the starting design.
    assert spacing.d.value == 80 * mm and collimating.f.value == 100 * mm
    assert float(simulation.run()[output].after.w) == pytest.approx(float(initial[output].after.w))

    report = str(solution)
    assert report.startswith("Solution: converged, feasible")
    for expected in ("Output radius", "Output divergence", "Total length", "Collimating lens aperture"):
        assert expected in report
    assert "VIOLATED" not in report
    assert {entry.label for entry in solution.constraints if entry.label} >= {
        "Total length", "Collimating lens aperture"}
    assert all(entry.satisfied for entry in solution.constraints)

    _, axes = optimized.plot()
    names = [label.get_text() for child in axes.child_axes for label in child.get_xticklabels()]
    assert names[0] == "Diverging lens" and names[-1] == "Output" and "Collimating lens" in names
    assert axes.get_ylabel() == "Beam radius ±w [mm]"


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
def test_focusing_design_puts_the_requested_waist_on_a_fixed_workpiece(method):
    """Every SciPy method must land on the same physical optimum."""
    spot = 30 * um
    source = GaussianBeam.from_waist(w0=0.75 * mm, wavelength=WAVELENGTH)
    target_focus = GaussianBeam.from_waist(w0=spot, wavelength=WAVELENGTH).numeric()
    input_gap = Space(d=60 * mm, label="Source to lens").variable(d=(40 * mm, None))
    lens = ThinLens(f=45 * mm, label="Focusing lens").variable(f=(35 * mm, 110 * mm))
    working_distance = Space(d=150 * mm - input_gap.d, label="Working distance")
    workpiece = Plane(label="Workpiece")
    system = System(input_gap, lens, working_distance, workpiece, optical_input=source)
    system.require(working_distance.d >= 50 * mm, label="Minimum working distance")

    simulation = system.build()
    sim = simulation.symbolic
    solver = Solver(simulation)
    solver.target(value=sim[workpiece].after.w, target=spot, label="Spot radius")
    solver.target(value=sim[workpiece].after.z0, target=0.0, scale=target_focus.z_r, label="Waist position")
    solver.require(sim[lens].after.divergence_angle <= 30 * mrad, label="Paraxial limit")
    options = {"ftol": 1e-14} if method == "SLSQP" else {"gtol": 1e-10, "xtol": 1e-12, "maxiter": 500}
    solution = solver.solve(method=method, options=options)
    focus = solution.run()[workpiece].after

    assert solution.success and solution.feasible
    assert float(focus.w) == pytest.approx(spot, rel=1e-3)
    assert abs(float(focus.z0)) < 1e-2 * float(target_focus.z_r)
    assert float(focus.w0) == pytest.approx(spot, rel=1e-3)
    # The workpiece did not move, and the working distance is what is left of the 150 mm.
    assert float(solution.run().z[-1]) == pytest.approx(150 * mm, rel=1e-12)
    assert solution.value(working_distance.d) == pytest.approx(150 * mm - solution.value(input_gap.d), rel=1e-12)
    assert solution.value(working_distance.d) >= 50 * mm * (1 - 1e-9)
    # A beam focused to a waist w0 by a thin lens at its waist-side focal plane: w0 ≈ λ f / (π w_lens).
    lens_radius = float(solution.run()[lens].after.w)
    assert solution.value(lens.f) == pytest.approx(np.pi * spot * lens_radius / WAVELENGTH, rel=2e-2)
    assert solution.iterations is not None and solution.iterations > 0


class PackagedLens(CompositeElement):
    """A thin lens centred in a housing of fixed length."""

    f: Node

    def __init__(self, f, length, label):
        super().__init__(label)
        self.f = f
        self.front = Space(d=length / 2)
        self.lens = ThinLens(f=self.f, label=f"{label} optic")
        self.back = Space(d=length / 2)


def test_fiber_coupling_design_with_custom_composites_reaches_the_target_mode():
    source = GaussianBeam.from_fiber_tip(wavelength=1550 * nm, MFD=10.4 * um)
    target_mode = GaussianBeam.from_fiber_tip(wavelength=1550 * nm, MFD=9.2 * um).numeric()
    gaps = [Space(d=5 * mm, label=name).variable("d") for name in ("Input gap", "Lens spacing", "Output gap")]
    lenses = [PackagedLens(f=5 * mm, length=4 * mm, label=name).variable(f=(4 * mm, 15 * mm))
              for name in ("Lens 1", "Lens 2")]
    fiber = Plane(label="Target fiber")
    system = System(gaps[0], lenses[0], gaps[1], lenses[1], gaps[2], fiber, optical_input=source)
    package_length = sum(element.element_length for element in system.elements)
    system.require(package_length <= 30 * mm, label="Package length")

    simulation = system.build()
    before = simulation.run()[fiber].after.overlap_with(target_mode)
    sim = simulation.symbolic
    solver = Solver(simulation)
    solver.target(value=sim[fiber].after.w, target=target_mode.w, label="Mode radius")
    solver.target(value=sim[fiber].after.z0, target=0.0, scale=target_mode.z_r, label="Waist position")
    for package in lenses:
        solver.require(sim[package.lens].after.w <= 1 * mm, label=f"{package.label} aperture")
    solution = solver.solve(options={"maxiter": 400, "ftol": 1e-13})
    arriving = solution.run()[fiber].after

    assert solution.success and solution.feasible
    assert float(before) < 0.2
    assert float(arriving.overlap_with(target_mode)) > 0.999
    assert float(arriving.w) == pytest.approx(float(target_mode.w), rel=1e-2)
    assert float(solution.run().z[-1]) <= 30 * mm * (1 + 1e-9)
    assert all(solution.value(gap.d) >= -1e-12 for gap in gaps)                    # gaps cannot go negative
    assert all(4 * mm - 1e-12 <= solution.value(package.f) <= 15 * mm + 1e-12 for package in lenses)
    # Each housing keeps its fixed length, and results address the optic inside it.
    result = solution.run()
    for package in lenses:
        assert float(result[package.back].z_end - result[package.front].z_start) == pytest.approx(4 * mm)
        assert units.format(result[package.lens].after.w, mm).endswith(" mm")


# -----------
# RAY DESIGNS
# -----------
def thick_lens_matrix(thickness, index, front_radius, back_radius):
    """Ray-transfer matrix of a thick lens in air for [x, theta], by hand."""
    def refraction(n_in, n_out, radius):
        return np.array([[1.0, 0.0], [(n_in - n_out) / (n_out * radius), n_in / n_out]])

    drift = np.array([[1.0, thickness], [0.0, 1.0]])
    return refraction(index, 1.0, back_radius) @ drift @ refraction(1.0, index, front_radius)


def test_imaging_design_finds_the_image_plane_of_a_thick_lens_placed_by_its_centre():
    thickness, index, front_radius, back_radius = 8 * mm, 1.5, 60 * mm, -90 * mm
    object_distance, lens_centre = 50 * mm, 150 * mm
    # Rays leaving one axial object point 50 mm before the start of the system.
    source = RayBundle.converging(radius=2 * mm, count=9, focal_distance=-object_distance)
    lens = ThickLens(d=thickness, n=index, R1=front_radius, R2=back_radius, label="Objective")
    image = Plane(label="Image plane")
    image_z = Parameter(400 * mm, name="Image position").variable(bounds=(200 * mm, 900 * mm))
    system = System(optical_input=source)
    system.add(lens, z=lens_centre, at="center")
    system.add(image, z=image_z)

    simulation = system.build()
    solver = Solver(simulation)
    solver.target(lambda sim: sim[image].after.x / mm, label="Ray heights at the image")
    solution = solver.solve(options={"ftol": 1e-15})
    result = solution.run()

    # Independent expectation: object distance s to the front vertex, then the image
    # distance behind the back vertex from the lens matrix [[A, B], [C, D]].
    front_vertex = lens_centre - thickness / 2
    s = object_distance + front_vertex
    (A, B), (C, D) = thick_lens_matrix(thickness, index, front_radius, back_radius)
    expected_image = front_vertex + thickness - (A * s + B) / (C * s + D)

    assert solution.success
    assert float(result[lens.front].z_start) == pytest.approx(front_vertex, rel=1e-12)
    assert solution.value(image_z) == pytest.approx(expected_image, rel=1e-6)
    assert float(result[image].after.rms_x) < 1e-9
    # The magnification of the image heights follows the same matrix: m = A + L C.
    L = expected_image - (front_vertex + thickness)
    np.testing.assert_allclose(result[image].after.theta,
                               np.asarray(result.source.theta) * (C * s + D), rtol=1e-6, atol=1e-12)
    assert A + L * C == pytest.approx(1.0 / (C * s + D), rel=1e-9)


# -----------------
# WHEN IT GOES WRONG
# -----------------
def focusing_problem():
    source = GaussianBeam.from_waist(w0=0.75 * mm, wavelength=WAVELENGTH)
    gap = Space(d=60 * mm, label="Source to lens").variable(d=(40 * mm, 100 * mm))
    lens = ThinLens(f=45 * mm, label="Focusing lens").variable(f=(35 * mm, 110 * mm))
    workpiece = Plane(label="Workpiece")
    system = System(gap, lens, Space(d=150 * mm - gap.d), workpiece, optical_input=source)
    return system.build(), lens, workpiece


def add_focus_goals(solver, simulation, workpiece, spot=30 * um):
    sim = simulation.symbolic
    z_r = GaussianBeam.from_waist(w0=spot, wavelength=WAVELENGTH).numeric().z_r
    solver.target(value=sim[workpiece].after.w, target=spot, label="Spot radius")
    solver.target(value=sim[workpiece].after.z0, target=0.0, scale=z_r, label="Waist position")
    return solver


def test_an_impossible_requirement_is_reported_as_infeasible_and_named():
    simulation, lens, workpiece = focusing_problem()
    solver = add_focus_goals(Solver(simulation), simulation, workpiece)
    # Diffraction puts the smallest spot this lens range can make far above 1 um.
    solver.require(simulation.symbolic[workpiece].after.w <= 1 * um, label="Sub-micron spot")
    solution = solver.solve(options={"maxiter": 200})

    assert not solution.feasible
    assert [entry.label for entry in solution.violations] == ["Sub-micron spot"]
    violated = solution.violations[0]
    assert float(violated.value) > 1 * um and not violated.satisfied and float(violated.violation) > 0
    report = str(solution)
    assert "infeasible (1 violated)" in report and "VIOLATED" in report
    # The result is still a usable design that can be inspected.
    assert np.isfinite(float(solution.run()[workpiece].after.w))
    assert 35 * mm <= solution.value(lens.f) <= 110 * mm


def test_a_stopped_optimization_can_be_continued_from_its_solution():
    simulation, lens, workpiece = focusing_problem()
    stopped = add_focus_goals(Solver(simulation), simulation, workpiece).solve(options={"maxiter": 2})

    assert not stopped.success and stopped.feasible
    assert "teration" in stopped.message
    first_cost = stopped.cost

    # Continue from where it stopped: the solution's simulation starts at the reached design.
    continued_simulation = stopped.simulation
    continued = add_focus_goals(Solver(continued_simulation), continued_simulation, workpiece).solve(
        options={"ftol": 1e-14})

    assert continued.success and continued.feasible
    assert continued.cost < first_cost
    assert float(continued.run()[workpiece].after.w) == pytest.approx(30 * um, rel=1e-3)
    assert continued_simulation.initial_values[lens.f] == pytest.approx(stopped.value(lens.f))
