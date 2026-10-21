"""Explicitly selected Autograd, independent of the parent process's backend."""
import numpy as np
import pytest
from tests.helpers.backend import assert_close


def test_numerical_operations_and_compiled_alias_snapshot(autograd_results):
    result = autograd_results
    assert result["backend"] == "autograd"
    assert result["metadata"] == {"variables": 2, "symbols": 1}
    residual = 1.3*(0.6**3+(-0.2)**2) + 0.5
    np.testing.assert_allclose(result["values"]["graph"]["0"]["values"],
                               [residual, np.sin(residual), 0.4], rtol=1e-13, atol=1e-14)


def test_complex_gradient_hessian_and_fourth_derivative(autograd_results):
    values = autograd_results["values"]
    np.testing.assert_allclose(values["complex"]["value"], 0.16, atol=1e-14)
    np.testing.assert_allclose(values["complex"]["gradient"], [0.96, -0.32], atol=1e-13)
    np.testing.assert_allclose(values["complex"]["hessian"], [[4.48, -0.96], [-0.96, 1.92]], atol=1e-13)
    assert values["higher"]["fourth_polynomial"] == pytest.approx(90.0)
    assert np.all(np.isfinite(values["higher"]["fourth_graph"]))
    assert np.all(np.isfinite(values["higher"]["nested"]))


def test_jit_identity_and_decorator(autograd_results):
    assert autograd_results["jit_identity"] == {"function": True, "decorator": True}
    for name, values in autograd_results["values"]["jit"].items():
        assert_close(values["compiled"], values["eager"], description=f"Autograd jit/{name}",
                     rtol=0, atol=1e-15)
    decorated = autograd_results["values"]["decorator"]
    np.testing.assert_allclose(decorated["values"], [8.0, 6.0])
    for name in ("jacobian", "jacrev"):
        np.testing.assert_allclose(decorated[name], [[12.0, 0.0], [3.0, 2.0]])


def test_require_checks_concrete_conditions(autograd_results):
    assert autograd_results["require"] == {
        "python_true": None, "numerical_true": None,
        "python_false": "deliberate validation failure",
        "numerical_false": "deliberate validation failure",
    }


@pytest.mark.parametrize("point", ["0", "1", "2"])
def test_forward_reverse_jacobians_agree(autograd_results, point):
    value = autograd_results["values"]["graph"][point]
    assert_close(value["jacobian"], value["jacrev"], description=f"Autograd jacobians/{point}",
                 rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", ["ray", "bundle"])
def test_builtin_ray_states_have_independent_abcd_expectation(autograd_results, kind):
    x, angle = (np.array(0.001), np.array(0.02)) if kind == "ray" else (
        np.array([0.001, -0.002]), np.array([0.02, -0.01]))
    actual = autograd_results["values"]["optics"][kind]
    for index, (distance, focal) in enumerate(((0.03, 0.12), (0.025, 0.10), (0.04, 0.15))):
        out_angle = angle - (x + distance*angle)/focal
        expected = np.stack((x + distance*angle + 0.04*out_angle, out_angle))
        assert_close(actual[index], expected, description=f"{kind}/{index}", rtol=1e-13, atol=1e-14)


def test_solver_objective_constraints_and_derivatives(autograd_results):
    for point, values in autograd_results["values"]["solver"].items():
        assert np.shape(values["gradient"]) == (2,)
        assert np.shape(values["constraints"]) == (2,)
        assert np.shape(values["jacobian"]) == (2, 2)
        for name in ("hessian", "weighted_hessian"):
            assert_close(values[name], np.asarray(values[name]).T,
                         description=f"Autograd/{point}/{name} symmetry", rtol=1e-11, atol=1e-10)


def test_gaussian_properties_against_independent_formula(autograd_results):
    wavelength, initial_q = 1.064e-6, 0.015+0.15j
    target_q = 1j*np.pi*0.0004**2/wavelength
    for index, (distance, focal) in enumerate(((0.03, 0.12), (0.025, 0.10), (0.04, 0.15))):
        q = (initial_q+distance)/(1-(initial_q+distance)/focal) + 0.04
        inv_q = 1/q
        w0 = np.sqrt(q.imag*wavelength/np.pi)
        overlap = 4*inv_q.imag*(1/target_q).imag/abs(np.conj(inv_q)-1/target_q)**2
        expected = [q.real, q.imag, np.sqrt(-wavelength/(np.pi*inv_q.imag)),
                    1/inv_q.real, q.imag, -q.real, w0,
                    np.arctan2(q.real, q.imag), wavelength/(np.pi*w0), overlap, distance+0.04]
        assert_close(autograd_results["values"]["optics"]["gaussian"][index], expected,
                     description=f"Gaussian physical formulas/{index}", rtol=1e-12, atol=1e-13)
