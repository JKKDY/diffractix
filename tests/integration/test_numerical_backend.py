"""Complete optical callbacks, including JIT and nested derivatives."""
import numpy as np
import pytest

from diffractix import GaussianBeam, Parameter, Solver, Space, System
from diffractix.backend import grad, hessian, jacobian, jit, xp
from diffractix.solver import Constraint
from diffractix.solver import solver as solver_module


def test_whole_solver_callbacks_jit_and_nested_derivatives(monkeypatch):
    distance = Parameter(0.03).variable(bounds=(0.01, 0.06))
    space = Space(d=distance)
    simulation = System(space, input=GaussianBeam.from_waist(0.0005, 1.064e-6)).build()
    solver = Solver(simulation)
    solver.target(lambda ctx: ctx.after(space).w * 1000)
    solver.require(Constraint(lambda ctx: ctx.after(space).w * 1000,
                              lower_bound=0.1, upper_bound=2.0))
    captured = {}

    class Captured(Exception):
        pass

    def capture(problem, **kwargs):
        captured['problem'] = problem.unscale()
        raise Captured

    monkeypatch.setattr(solver_module, 'solve_scipy', capture)
    with pytest.raises(Captured):
        solver.solve()
    problem = captured['problem']
    theta = xp.array([0.03])
    for function in (problem.objective, problem.gradient, problem.constraints,
                     problem.jacobian, problem.objective_hessian,
                     hessian(lambda t: xp.sum(grad(problem.objective)(t)**2))):
        eager = function(theta)
        compiled = jit(function)(theta)
        assert np.all(np.isfinite(eager))
        assert np.all(np.isfinite(compiled))
        np.testing.assert_allclose(compiled, eager, rtol=2e-10, atol=1e-12)

    # Independent paraxial free-space formula, not another compiled execution.
    k = 1.064e-6 / (np.pi * 0.0005)
    np.testing.assert_allclose(problem.objective(theta),
                               1e6 * (0.0005**2 + (k * theta[0])**2), rtol=1e-12)
    np.testing.assert_allclose(problem.gradient(theta), [2e6 * k*k * 0.03], rtol=1e-12)
    np.testing.assert_allclose(problem.objective_hessian(theta), [[2e6*k*k]], rtol=1e-12)
    multipliers = xp.array([2.0])
    np.testing.assert_allclose(jit(problem.constraint_hessian)(theta, multipliers),
                               problem.constraint_hessian(theta, multipliers), rtol=1e-11)


def test_gaussian_curvature_factories_are_traceable_at_waist():
    def curvature(t):
        beam = GaussianBeam(q=t[0] + 1j, wavelength=1e-6, n=1.0)
        return xp.stack((beam.w, beam.gouy_phase))
    theta = xp.array([0.0])
    for derivative in (jacobian(curvature), hessian(lambda t: xp.sum(curvature(t)**2))):
        actual = jit(derivative)(theta)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, derivative(theta), rtol=1e-12, atol=1e-12)
    for radius in (np.inf, 0.2, -0.2):
        value = jit(lambda r: GaussianBeam.from_w_and_R(0.001, r, 1e-6).w)(radius)
        assert value == pytest.approx(0.001)


def test_gaussian_concrete_domain_validation_remains_active():
    first = GaussianBeam(q=1j, wavelength=1e-6, n=1.0)
    with pytest.raises(ValueError, match='media'):
        first.overlap_with(GaussianBeam(q=1j, wavelength=1e-6, n=1.5))
    with pytest.raises(ValueError, match='wavelength'):
        first.overlap_with(GaussianBeam(q=1j, wavelength=2e-6, n=1.0))
    with pytest.raises(Exception, match='Invalid q'):
        GaussianBeam.w_R_from_q(-1j, 1e-6, 1.0)


def test_backend_derivative_wrappers_and_configured_jit_decorator():
    from diffractix.backend import jacrev

    @jit(static_argnums=1)
    def expression(theta, power):
        return xp.stack((theta[0]**power, theta[0] * theta[1]))

    theta = xp.array([2.0, 3.0])
    np.testing.assert_allclose(expression(theta, 3), [8.0, 6.0])
    function = lambda t: expression(t, 3)
    for derivative in (jacobian, jacrev):
        actual = derivative(function)(theta)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, [[12.0, 0.0], [3.0, 2.0]])


@pytest.mark.parametrize('kind', ['ray', 'bundle', 'custom'])
def test_jit_propagation_preserves_python_and_custom_states(kind):
    from dataclasses import dataclass
    from diffractix import ABCD, ParaxialRay, RayBundle
    from diffractix.beams.base import ParaxialState

    @dataclass
    class CustomState(ParaxialState):
        x: object
        theta: object

        def propagate(self, A, B, C, D, n):
            return CustomState(A*self.x + B*self.theta, C*self.x + D*self.theta)

    source = (ParaxialRay(x=0.001, theta=0.02) if kind == 'ray' else
              RayBundle(x=np.array([0.001, 0.002]), theta=np.array([0.02, -0.01]))
              if kind == 'bundle' else CustomState(0.001, 0.02))
    distance = Parameter(0.03).variable()
    simulation = System(ABCD(A=1, B=distance, C=0, D=1), input=source).build()

    def values(theta):
        result = simulation.run(theta)
        return xp.stack((result.final.x, result.final.theta))

    theta = xp.array([0.03])
    for function in (values, jacobian(values), hessian(lambda t: xp.sum(values(t)**2))):
        actual = jit(function)(theta)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, function(theta), rtol=1e-12, atol=1e-14)


@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part:numpy.exceptions.ComplexWarning"
)
def test_complex_optical_derivatives_against_central_differences():
    from diffractix import GaussianAperture, ThinLens

    distance = Parameter(0.03).variable()
    focal_length = Parameter(0.12).variable()
    simulation = System(Space(d=distance), ThinLens(f=focal_length),
                        GaussianAperture(a=0.001), Space(d=0.04),
                        input=GaussianBeam.from_waist(0.0005, 1.064e-6)).build()

    def values(theta):
        state = simulation.run(theta).final
        return xp.stack((1000*xp.real(state.w), xp.real(state.q), xp.imag(state.q)))

    def cost(theta):
        return xp.sum(values(theta)**2)

    theta = np.array([0.03, 0.12])
    step = 1e-6
    differences = np.eye(2) * step
    numerical_jacobian = np.stack([(values(theta+d) - values(theta-d))/(2*step)
                                   for d in differences], axis=1)
    numerical_gradient = np.array([(cost(theta+d)-cost(theta-d))/(2*step)
                                   for d in differences])
    numerical_hessian = np.stack([(grad(cost)(theta+d)-grad(cost)(theta-d))/(2*step)
                                  for d in differences], axis=1)
    for actual, expected in ((jacobian(values)(theta), numerical_jacobian),
                             (grad(cost)(theta), numerical_gradient),
                             (hessian(cost)(theta), numerical_hessian)):
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-8)
