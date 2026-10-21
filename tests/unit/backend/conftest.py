"""Each backend-specific suite gets results from an explicitly selected process."""
import pytest
from tests.helpers.backend import JAX_AVAILABLE, numerical_results


@pytest.fixture(scope="session")
def autograd_results():
    # Required dependency: absence is an environment failure, not a skip.
    return numerical_results("autograd")


@pytest.fixture(scope="session")
def jax_results():
    if not JAX_AVAILABLE:
        pytest.skip("Optional JAX numerical backend is not installed")
    return numerical_results("jax")
