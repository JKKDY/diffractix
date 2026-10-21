"""Identical functions/inputs, independent processes, host NumPy comparisons."""
import pytest
from tests.helpers.backend import JAX_AVAILABLE, assert_close

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="Parity requires optional JAX and required Autograd")


@pytest.mark.parametrize("point", ["0", "1", "2"])
@pytest.mark.parametrize("operation", ["values", "gradient", "jacobian", "jacrev", "hessian", "weighted_hessian"])
def test_compiled_graph_parity(autograd_results, jax_results, point, operation):
    # Higher derivatives amplify floating-point rounding more than primal values.
    rtol, atol = (2e-10, 1e-12) if operation == "values" else (2e-9, 1e-10)
    assert_close(jax_results["values"]["graph"][point][operation],
                 autograd_results["values"]["graph"][point][operation],
                 description=f"graph/theta[{point}]/{operation}", rtol=rtol, atol=atol)


@pytest.mark.parametrize("kind", ["gaussian", "ray", "bundle", "refractive", "aperture"])
def test_optical_values_parity(autograd_results, jax_results, kind):
    assert_close(jax_results["values"]["optics"][kind], autograd_results["values"]["optics"][kind],
                 description=f"optics/{kind} (three parameter vectors)", rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("point", ["0", "1", "2"])
@pytest.mark.parametrize("operation", ["objective", "constraints", "gradient", "jacobian", "hessian", "weighted_hessian"])
def test_real_solver_primitives_parity(autograd_results, jax_results, point, operation):
    # Physical-coordinate derivatives contain inverse optical lengths.
    rtol, atol = (2e-10, 1e-12) if operation in ("objective", "constraints") else (2e-9, 1e-8)
    assert_close(jax_results["values"]["solver"][point][operation],
                 autograd_results["values"]["solver"][point][operation],
                 description=f"solver/theta[{point}]/{operation}", rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", ["array", "complex", "higher", "scalar_capture"])
def test_complex_array_and_nested_derivative_parity(autograd_results, jax_results, case):
    assert_close(jax_results["values"][case], autograd_results["values"][case],
                 description=case, rtol=2e-9, atol=1e-8)
