"""Build/solve snapshots stay independent of declarative model edits."""
import numpy as np
import pytest
from diffractix import Constraint, Parameter, Plane, Ray, Solver, Space, System


@pytest.mark.parametrize("owner", ["system", "element"])
def test_built_constraint_program_retains_handle_before_external_reassignment(owner):
    distance = Parameter(0.05).variable(bounds=(0.01, 0.3))
    space = Space(d=distance)
    system = System(space, optical_input=Ray(x=0, theta=0.01, n=1))
    getattr(system if owner == "system" else space, "require")(
        Constraint(space.d, upper_bound=0.1, label="original limit"))
    simulation = system.build()
    space.d = 0.25
    distance.value = 0.2
    solver = Solver(simulation).target(lambda sim: sim.final.x / 0.002 - 1)
    solution = solver.solve(options={"ftol": 1e-12})
    assert solution.success and solution.feasible
    assert solution.value(distance) == pytest.approx(0.1, abs=1e-7)
    assert solution.run().final.x == pytest.approx(0.001, abs=1e-9)
    assert simulation.initial_values.values[0] == 0.05
    assert [entry.label for entry in solution.constraints].count("original limit") == 1


def test_source_and_model_mutation_cannot_change_an_existing_solution():
    position = Parameter(0.5).variable(bounds=(-2, 2))
    source = Ray(x=position, theta=0.02, n=1)
    space = Space(d=0.1)
    solver = Solver(System(space, optical_input=source)).target(lambda sim: sim.final.x - 1)
    solution = solver.solve()
    initial_result = solution.run()
    source.theta = 0.9
    source.x = -10
    position.value = -1
    space.d = 10
    np.testing.assert_allclose(solution.simulation.run().final.vector, [1, 0.02], atol=1e-7)
    np.testing.assert_allclose(initial_result.final.vector, [1, 0.02], atol=1e-7)
    assert solution.run() is initial_result
    with pytest.raises(RuntimeError, match="fixed"):
        solver.target(position)
    with pytest.raises(RuntimeError, match="fixed"):
        solver.require(position >= 0)


def _weak_references_after_dropping_the_model():
    """Build, solve, drop every user reference, and report what the results still keep alive."""
    import gc
    import weakref

    distance = Parameter(0.05).variable(bounds=(0.01, 0.2))
    gap = Space(d=distance)
    system = System(gap, optical_input=Ray(x=0, theta=0.01, n=1)).add(Plane(), z=0.1)
    simulation = system.build()
    solver = Solver(simulation).target(distance / 0.08 - 1)
    solution = solver.solve(options={"ftol": 1e-12})
    references = {"parameter": weakref.ref(distance), "element": weakref.ref(gap)}
    del distance, gap, system, solver
    gc.collect()
    return simulation, solution, references


def test_simulation_keeps_the_objects_whose_ids_index_its_metadata_alive():
    """ids are only unique among live objects, so a freed model object could be mistaken
    for a new one. The metadata must therefore hold what it indexes."""
    simulation, _, references = _weak_references_after_dropping_the_model()
    assert references["parameter"]() is not None
    assert references["element"]() is not None
    assert simulation.parameter_data[0].parameter_id == id(references["parameter"]())
    assert any(data.element_id == id(references["element"]()) for data in simulation.element_data)


def test_solution_simulation_and_a_new_solver_keep_the_same_objects_alive():
    import gc

    simulation, solution, references = _weak_references_after_dropping_the_model()
    del simulation
    gc.collect()
    assert references["parameter"]() is not None and references["element"]() is not None
    assert solution.simulation.parameter_data[0].parameter is references["parameter"]()
    reused = Solver(solution.simulation)
    del solution
    gc.collect()
    assert references["parameter"]() is not None and references["element"]() is not None
    assert reused.simulation.initial_values[references["parameter"]()] == pytest.approx(0.08, abs=1e-6)
