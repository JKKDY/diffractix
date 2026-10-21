"""Reuse the real advanced design; evaluate, but do not run optimization."""
import pytest
from tests.helpers.backend import JAX_AVAILABLE, assert_close, numerical_results

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="Advanced parity requires optional JAX")


@pytest.fixture(scope="module")
def advanced_pair():
    return (numerical_results("autograd", "advanced")["values"],
            numerical_results("jax", "advanced")["values"])


@pytest.mark.parametrize("point", ["0", "1", "2"])
@pytest.mark.parametrize("operation", ["radii", "objective", "constraints", "gradient", "jacobian"])
def test_multistation_backend_parity(advanced_pair, point, operation):
    autograd, jax = advanced_pair
    # Gradients are in inverse-metre coordinates; their absolute scale is larger.
    rtol, atol = (2e-10, 1e-12) if operation in ("radii", "objective", "constraints") else (2e-9, 2e-9)
    assert_close(jax[point][operation], autograd[point][operation],
                 description=f"multistation/theta[{point}]/{operation}", rtol=rtol, atol=atol)
