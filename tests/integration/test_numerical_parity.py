"""Current graph, optical propagation and real inverse-design backend parity."""
import numpy as np
import pytest
from tests.helpers.backend import JAX_AVAILABLE, numerical_results, assert_close

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="Cross-backend parity needs optional JAX")


@pytest.fixture(scope="module")
def pair():
    return tuple(numerical_results(name, "integration")["values"] for name in ("autograd", "jax"))


@pytest.mark.parametrize("group", ["graph", "optics", "solve"])
def test_backend_parity_and_independent_physical_expectations(pair, group):
    autograd, jax = pair
    if group == "solve":
        for name in ("success", "feasible"):
            assert jax[group][name] == autograd[group][name]
        for name in ("x", "cost", "final_x"):
            assert_close(jax[group][name], autograd[group][name], description=f"solve/{name}", rtol=2e-9, atol=1e-11)
    else:
        assert_close(jax[group], autograd[group], description=group, rtol=2e-9, atol=1e-11)
    for values in pair:
        if group == "graph":
            data = values[group]
            np.testing.assert_allclose(data["values"], [0.067, 0.64], atol=1e-14)
            expected = np.zeros((2, 2))
            expected[:, data["p_index"]] = [0.27, -0.2]
            expected[:, data["q_index"]] = [-0.4, 0.3]
            np.testing.assert_allclose(data["jacobian"], expected, atol=1e-14)
        elif group == "optics":
            wavelength, waist, z = 1.064e-6, 0.0005, 0.03
            rayleigh = np.pi * waist**2 / wavelength
            np.testing.assert_allclose(values[group]["values"],
                [waist*np.sqrt(1+(z/rayleigh)**2), z, rayleigh], rtol=1e-12)
        else:
            assert values[group]["success"] and values[group]["feasible"]
            np.testing.assert_allclose(values[group]["x"], [0.002], atol=1e-9)
            assert values[group]["final_x"] == pytest.approx(0.003, abs=1e-9)
            assert values[group]["cost"] < 1e-12
