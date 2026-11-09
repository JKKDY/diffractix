"""Small optical inverse designs with independent analytical optima."""
import numpy as np
import pytest
from diffractix import Constraint, GaussianBeam, Interface, Parameter, Plane, Ray, RayBundle, Slab, Solver, Space, System, ThinLens


@pytest.mark.parametrize("use_jit", [False, True])
def test_source_angle_and_lens_focal_length_recover_known_ray_design(use_jit):
    angle = Parameter(0.01).variable(bounds=(-0.1, 0.1))
    focal = Parameter(0.15).variable(bounds=(0.05, 0.3))
    lens = ThinLens(f=focal)
    system = System(Space(d=0.1), lens, optical_input=Ray(x=0.002, theta=angle, n=1))
    solver = Solver(system)
    solver.target(value=lambda sim: sim[lens].before.x, target=0.004, scale=0.004, label="lens height")
    solver.target(value=lambda sim: sim.final.theta, target=0, scale=0.02, label="collimated")
    solution = solver.solve(jit=use_jit, options={"ftol": 1e-12, "maxiter": 100})
    assert solution.success and solution.feasible
    assert solution.value(angle) == pytest.approx(0.02, abs=2e-7)
    assert solution.value(lens.f) == pytest.approx(0.2, abs=2e-6)
    assert solution.run().final.theta == pytest.approx(0, abs=2e-8)
    assert solution.cost < 1e-11
    assert sum(entry.cost for entry in solution.objectives) == pytest.approx(solution.cost)


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
def test_gaussian_waist_position_is_optimized_through_real_propagation(method):
    distance = Parameter(0.04).variable(bounds=(0.01, 0.15))
    space = Space(d=distance)
    solver = Solver(System(space, optical_input=GaussianBeam(q=-0.08+0.03j, wavelength=1e-6, n=1)))
    solver.target(lambda sim: sim.final.z0 / 0.08)
    solution = solver.solve(method=method, options={"maxiter": 150})
    assert solution.success and solution.feasible
    assert solution.iterations > 0 and solution.function_evaluations > 0
    assert solution.value(distance) == pytest.approx(0.08, abs=2e-6)
    assert solution.run().final.q == pytest.approx(0.03j, abs=2e-6)
    assert solution.run().final.w == pytest.approx(np.sqrt(1e-6 * 0.03 / np.pi), rel=1e-7)


def test_system_composite_element_and_solver_constraints_change_physical_optimum():
    thickness = Parameter(0.02).variable(bounds=(0.005, 0.05))
    slab = Slab(d=thickness, n=1.5)
    slab.require(Constraint(slab.d, upper_bound=0.03, label="composite maximum"))
    slab.body.require(Constraint(slab.body.d, lower_bound=0.01, label="element minimum"))
    system = System(slab, optical_input=Ray(x=0, theta=0.03, n=1))
    system.require(Constraint(thickness, upper_bound=0.025, label="system maximum"))
    solver = Solver(system)
    solver.target(value=lambda sim: sim.final.x, target=0.001, scale=0.001)
    solver.require(Constraint(lambda sim: sim.final.x, upper_bound=0.0004, label="output limit"))
    solution = solver.solve(options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    # Two planar interfaces conserve n*theta: glass angle = 0.03/1.5.
    assert solution.value(thickness) == pytest.approx(0.02, abs=1e-7)
    assert solution.run().final.x == pytest.approx(0.0004, abs=2e-9)
    labels = [entry.label for entry in solution.constraints]
    assert labels.index("system maximum") < labels.index("composite maximum") < labels.index("element minimum") < labels.index("output limit")
    assert solution.run().final.n == pytest.approx(1)


def test_independent_indices_remain_continuous_during_optimization():
    upstream = Parameter(1.2).variable(bounds=(1, 2))
    incoming = Parameter(1.2).variable(bounds=(1, 2))
    interface = Interface(n1=incoming, n2=1)
    solver = Solver(System(interface, optical_input=Ray(x=0, theta=0.01, n=upstream)))
    solver.target(upstream - 1.6, incoming - 1.1)
    solution = solver.solve(options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(upstream) == pytest.approx(1.35, abs=1e-6)
    assert solution.value(incoming) == pytest.approx(1.35, abs=1e-6)
    continuity = [entry for entry in solution.constraints if entry.lower_bound == entry.upper_bound == 0]
    assert len(continuity) == 1
    assert abs(continuity[0].value) < 1e-9


def test_repeated_space_occurrence_traces_longitudinal_and_state_quantities():
    distance = Parameter(0.03).variable(bounds=(0.01, 0.2))
    repeated = Space(d=distance)
    solver = Solver(System(repeated, repeated, optical_input=Ray(x=0, theta=0.02, n=1)))
    solver.target(lambda sim: sim[repeated, 1].z_end / 0.2 - 1)
    solver.require(Constraint(lambda sim: sim[repeated, 0].after.x, upper_bound=0.003))
    solution = solver.solve()
    assert solution.success and solution.feasible
    assert solution.value(distance) == pytest.approx(0.1, abs=1e-7)
    result = solution.run()
    assert result[repeated, 1].after.x == pytest.approx(0.004, abs=1e-8)
    assert result[repeated, 1].z_start == pytest.approx(0.1, abs=1e-7)
    with pytest.raises(ValueError, match="occurrence"):
        result[repeated]


def test_generated_gap_constraint_prevents_optimized_element_crossing_station():
    distance = Parameter(0.05).variable(bounds=(0.01, 0.2))
    upstream, station = Space(d=distance), Plane()
    system = System(upstream, optical_input=Ray(x=0, theta=0.01, n=1))
    system.add(station, z=0.1)
    solver = Solver(system).target(distance / 0.2 - 1)
    solution = solver.solve(options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(distance) == pytest.approx(0.1, abs=1e-7)
    assert solution.run()[station].z_start == pytest.approx(0.1)
    assert solution.run().z[-1] == pytest.approx(0.1)
    assert min(np.diff(solution.run().z)) >= -1e-9


def test_coupled_optical_gaps_keep_mechanical_clearance_at_small_scale():
    first = Parameter(0.006).variable(bounds=(0.004, 0.01))
    middle = Parameter(0.008).variable(bounds=(0.005, 0.014))
    output = Space(d=0.025 - first - middle)
    solver = Solver(System(Space(d=first), Space(d=middle), output,
                           optical_input=Ray(x=0, theta=0.01, n=1)))
    solver.target((first - 0.01) / 0.01, (middle - 0.014) / 0.01)
    solver.require(Constraint(output.d, lower_bound=0.004, label="output clearance"))
    solution = solver.solve(options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(first) == pytest.approx(0.0085, abs=1e-7)
    assert solution.value(middle) == pytest.approx(0.0125, abs=1e-7)
    result = solution.run()
    assert result[output].z_end - result[output].z_start == pytest.approx(0.004, abs=1e-8)
    assert result.final.x == pytest.approx(0.00025, abs=1e-9)
    assert any(entry.label == "output clearance" for entry in solution.constraints)


@pytest.mark.parametrize("use_jit", [False, True])
def test_ray_bundle_array_residual_recovers_analytical_focus(use_jit):
    length, radius = 0.1, 0.001
    positions = np.array([-radius, 0., radius])
    focal = Parameter(0.15).variable(bounds=(0.05, 0.3))
    solver = Solver(System(ThinLens(f=focal), Space(d=length),
                           optical_input=RayBundle(x=positions, theta=np.zeros(3), n=1)))
    solver.target(lambda sim: sim.final.x / radius, label="bundle focus")
    solution = solver.solve(jit=use_jit, options={"ftol": 1e-12, "maxiter": 100})
    assert solution.success and solution.feasible
    f = solution.value(focal)
    assert f == pytest.approx(length, abs=1e-7)
    final = solution.run().final
    np.testing.assert_allclose(final.x, positions * (1-length/f), atol=1e-12)
    np.testing.assert_allclose(final.theta, -positions/f, atol=1e-12)
    np.testing.assert_allclose(final.x, np.zeros(3), atol=1e-9)
    objective, = solution.objectives
    assert objective.value.shape == (3,)
    np.testing.assert_allclose(objective.value, final.x/radius, atol=1e-12)
    assert objective.cost == pytest.approx(np.sum((positions/radius*(1-length/f))**2), abs=1e-14)
    assert solution.cost == pytest.approx(objective.cost, abs=1e-14)


@pytest.mark.parametrize("use_jit", [False, True])
def test_ray_bundle_array_target_and_scale_recover_nonzero_image(use_jit):
    positions = np.array([-0.001, 0., 0.001])
    target = positions / 2
    scale = np.array([0.001, 0.002, 0.003])
    focal = Parameter(0.15).variable(bounds=(0.06, 0.3))
    solver = Solver(System(ThinLens(f=focal), Space(d=0.05),
                           optical_input=RayBundle(x=positions, theta=np.zeros(3), n=1)))
    solver.target(value=lambda sim: sim.final.x, target=target, scale=scale, label="image")
    solution = solver.solve(jit=use_jit, options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    # x_out = x_in*(1-L/f) = x_in/2 requires f=2L.
    assert solution.value(focal) == pytest.approx(0.1, abs=2e-7)
    np.testing.assert_allclose(solution.run().final.x, target, atol=2e-9)
    objective, = solution.objectives
    np.testing.assert_allclose(objective.value, (solution.run().final.x-target)/scale, atol=1e-12)
    assert objective.cost == pytest.approx(solution.cost, abs=1e-14)
    assert solution.cost < 1e-11




def test_quick_start_workflow_focuses_a_ray_onto_the_axis():
    """The README workflow on the current API: absolute placements, a bounded lens, one target."""
    ray = Ray(x=0.01, theta=0.0)
    lens = ThinLens(f=0.2)
    lens.f.variable(bounds=(0.05, 0.5))
    image_plane = Plane(label="Image plane")
    system = System(optical_input=ray)
    system.add(lens, z=0.1)
    system.add(image_plane, z=0.2)
    simulation = system.build()
    initial = simulation.run()
    assert initial[image_plane].after.x == pytest.approx(0.005)

    solver = Solver(simulation)
    solver.target(value=lambda result: result[image_plane].after.x, target=0.0, scale=0.01)
    solution = solver.solve(method="SLSQP")

    # Default SLSQP tolerances: precise to the README's printed 0.1 mm.
    assert solution.success and solution.feasible
    assert solution.value(lens.f) == pytest.approx(0.1, abs=5e-5)
    assert solution.run()[image_plane].after.x == pytest.approx(0.0, abs=5e-6)
    assert lens.f.value == 0.2
