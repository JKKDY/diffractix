"""Source fields and element fields share canonical theta and snapshot values."""
import numpy as np
import pytest
from diffractix import ABCD, ComplexParameter, GaussianBeam, Parameter, Ray, RayBundle, Solver, Space, System
from diffractix.backend import grad,hessian,jacobian,jit,xp


def test_source_and_optical_fields_share_one_parameter_and_derivatives():
    p=Parameter(0.02).variable()
    source=Ray(x=p,theta=0.01)
    simulation=System(Space(d=2*p),source=source).build()
    assert len(simulation.parameter_data)==1
    function=lambda theta:simulation.run(theta).final.x
    point=xp.array([0.03])
    assert function(point)==pytest.approx(0.0306)
    np.testing.assert_allclose(jit(grad(function))(point),[1.02])
    np.testing.assert_allclose(jit(hessian(lambda t:function(t)**2))(point),[[2*1.02**2]])
    p.value=0.5
    source.theta=0.9
    assert simulation.run().final.x==pytest.approx(0.0204)


def test_source_array_views_are_owned_by_build_and_previous_results():
    original=np.array([[-0.001,0.002],[0.01,-0.02]])
    source=RayBundle(x=original[0],theta=original[1])
    simulation=System(Space(d=0.1),source=source).build()
    first=simulation.run()
    expected=np.array(first.final.vectors)
    original[:]=99
    source.x=np.array([4.,5.])
    np.testing.assert_array_equal(first.final.vectors,expected)
    np.testing.assert_array_equal(simulation.run().final.vectors,expected)


def test_wavelength_snapshot_and_source_medium_are_consistent():
    wavelength=Parameter(1e-6).variable(bounds=(0.5e-6,2e-6))
    beam=GaussianBeam(q=0.02j,wavelength=wavelength,n=1.33)
    simulation=System(Space(d=0.03),source=beam).build()
    first=simulation.run()
    wavelength.value=2e-6
    beam.q=9j
    np.testing.assert_allclose(simulation.run().final.q,0.03+0.02j)
    assert first.final.wavelength==pytest.approx(1e-6)
    updated=simulation.run([1.5e-6])
    assert updated.final.wavelength==pytest.approx(1.5e-6)
    assert updated.final.n==pytest.approx(1.33)
    assert updated.final.w/first.final.w==pytest.approx(np.sqrt(1.5))



@pytest.mark.parametrize("q", [-1j, 1., 0.])
def test_build_rejects_source_q_mutated_after_construction(q):
    source = GaussianBeam(q=1j, wavelength=1e-6, n=1)
    source.q = q
    with pytest.raises(ValueError, match="Invalid q-parameter"):
        System(Space(d=0.1), source=source).build()


def test_complex_source_parameter_runs_and_solves_in_real_coordinates():
    q = ComplexParameter(0.02 + 0.04j, "source_q").variable(bounds=((-0.1, 0.1), (0.01, 0.2)))
    source = GaussianBeam(q=q, wavelength=1e-6)
    simulation = System(Space(d=0.03), source=source).build()
    real, imag = simulation.parameter_data
    assert real.parameter is q.real
    assert imag.parameter is q.imag
    group, = simulation.grouped_parameter_data
    assert group.value == q.value
    assert group.bounds == q.bounds

    guess = simulation.initial_values
    guess[source.q] = 0.01 + 0.03j
    assert guess[q] == 0.01 + 0.03j
    np.testing.assert_array_equal(guess.values, [0.01, 0.03])
    assert simulation.initial_values[q] == 0.02 + 0.04j
    assert simulation.run(guess.values).final.q == pytest.approx(0.04 + 0.03j)
    with pytest.raises(ValueError, match="positive imaginary part"):
        simulation.run([0.02, -0.01])

    def final_parts(theta):
        final = simulation.run(theta).final.q
        return xp.stack((xp.real(final), xp.imag(final)))

    np.testing.assert_allclose(jit(jacobian(final_parts))(guess.values), np.eye(2))
    solver = Solver(simulation)
    solver.target(value=simulation.symbolic.final.q.real, target=0.05)
    solver.target(value=simulation.symbolic.final.q.imag, target=0.06)
    solution = solver.solve(x0=guess)
    assert solution.success and solution.feasible, solution.message
    assert solution.value(source.q) == pytest.approx(0.02 + 0.06j, abs=1e-7)
    assert solution.value(q) == solution.simulation.initial_values[q]
    assert "source_q.real" not in str(solution)
    assert "source_q" in str(solution)
    assert "0.02+0.06j" in str(solution)


def test_complex_abcd_entry_uses_the_same_graph_parameters():
    b = ComplexParameter(0.02 + 0.01j).variable()
    simulation = System(ABCD(B=b), source=GaussianBeam(q=0.03j, wavelength=1e-6)).build()
    guess = simulation.initial_values
    np.testing.assert_array_equal(guess.values, [0.02, 0.01])
    assert simulation.run().final.q == pytest.approx(0.02 + 0.04j)
    assert jit(grad(lambda theta: xp.imag(simulation.run(theta).final.q)))(guess.values)[1] == pytest.approx(1)


def test_complex_bounds_are_atomic_and_literal_q_is_a_parameter():
    source = GaussianBeam(q=0.02 + 0.04j, wavelength=1e-6)
    q = source.q.node
    assert isinstance(q, ComplexParameter)
    old_bounds = q.bounds
    with pytest.raises(ValueError, match="outside bounds"):
        source.variable(q=((-0.1, 0.1), (0.05, 0.1)))
    assert q.bounds == old_bounds
    assert not q.real.is_variable and not q.imag.is_variable
    source.variable("q")
    assert q.real.is_variable and q.imag.is_variable
    source.fixed("q")
    assert not q.real.is_variable and not q.imag.is_variable
    source.q = 0.04 + 0.05j
    assert source.numeric({"ambient_n": 1.0}).q == pytest.approx(0.04 + 0.05j)


def test_complex_component_assignment_keeps_graph_identity_and_bounds():
    source = GaussianBeam(q=0.02 + 0.04j, wavelength=1e-6, n=1)
    q = source.q.node.variable(bounds=((-0.1, 0.1), (0.01, 0.2)))
    real, imag = q.real, q.imag
    bounds = q.bounds
    expression = 2 * source.q

    q.real = 0.01
    source.q.imag = 0.06

    assert q.real is real and q.imag is imag
    assert real.is_variable and imag.is_variable
    assert q.bounds == bounds
    assert q.value == 0.01 + 0.06j
    assert expression.evaluate() == pytest.approx(0.02 + 0.12j)
    assert source.numeric().q == pytest.approx(q.value)

    with pytest.raises(TypeError, match="real scalar"):
        q.imag = 1j
    with pytest.raises(ValueError, match="finite"):
        source.q.real = np.inf
    assert q.value == 0.01 + 0.06j


def test_complex_residual_requires_a_real_quantity():
    source = GaussianBeam(q=0.03j, wavelength=1e-6).variable("q")
    solver = Solver(System(source=source)).target(source.q)
    with pytest.raises(TypeError, match="Objective must be real-valued"):
        solver.solve()
