"""JAX-specific transformations; no reconfiguration of the pytest process."""
import numpy as np
import pytest
from tests.helpers.backend import JAX_AVAILABLE, assert_close, run_python

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="Optional JAX numerical backend is not installed")


def test_actual_jax_selection_and_64_bit_precision(jax_results):
    assert jax_results["backend"] == "jax"
    assert jax_results["jax"]["float64"] == "float64"
    assert jax_results["jit_identity"] == {"function": False, "decorator": False}


def test_explicit_python_scalar_can_be_captured_as_a_graph_constant(jax_results):
    values = jax_results["values"]["scalar_capture"]
    np.testing.assert_allclose(values["values"], [0.2], rtol=1e-13)
    np.testing.assert_allclose(values["gradient"], [1/0.15], rtol=1e-13)


def test_concrete_backend_scalar_arrays_can_be_captured_as_literal_constants():
    assert run_python('''
import numpy as np
from diffractix.backend import xp
from diffractix.graph import Parameter, compile_ast
p = Parameter(1.0).variable()
for scalar in (2, 2.0, 2.0 + 1j):
    graph = compile_ast((p + xp.array(scalar),))
    assert graph.variables == (p,)
    for theta in (1.0, 4.0):
        np.testing.assert_allclose(graph.evaluate(xp.array([theta])), [theta + scalar])
print('ok')
''', 'jax') == 'ok'


def test_backend_vector_arrays_are_not_converted_to_literal_constants():
    assert run_python('''
import pytest
from diffractix.backend import xp
from diffractix.graph import Parameter
p = Parameter(1.0).variable()
for value in (xp.array([2.0]), xp.array([2.0, 3.0])):
    with pytest.raises(TypeError, match='numeric scalar'):
        p + value
print('ok')
''', 'jax') == 'ok'


def test_active_jax_tracers_cannot_be_captured_as_literal_constants():
    assert run_python('''
import pytest
from jax.errors import ConcretizationTypeError
from diffractix.backend import jit, xp
from diffractix.graph import Parameter
p = Parameter(1.0).variable()
with pytest.raises(ConcretizationTypeError, match='concrete'):
    jit(lambda value: p + value)(xp.array(2.0))
print('ok')
''', 'jax') == 'ok'


@pytest.mark.parametrize("name", [
    "graph", "graph_gradient", "graph_jacobian", "objective", "gradient",
    "constraints", "jacobian", "hessian", "weighted_hessian", "nested", "fourth",
    "gaussian", "ray", "bundle", "refractive", "aperture", "input",
])
def test_complete_jit_callbacks_and_intermediate_state_objects(jax_results, name):
    values = jax_results["values"]["jit"][name]
    assert_close(values["compiled"], values["eager"], description=f"JAX jit/{name}",
                 rtol=2e-10, atol=1e-10)


def test_jacfwd_jacrev_hessian_and_higher_derivatives(jax_results):
    decorated = jax_results["values"]["decorator"]
    np.testing.assert_allclose(decorated["values"], [8.0, 6.0])
    for name in ("jacobian", "jacrev"):
        np.testing.assert_allclose(decorated[name], [[12.0, 0.0], [3.0, 2.0]])
    for point, values in jax_results["values"]["graph"].items():
        assert_close(values["jacobian"], values["jacrev"], description=f"JAX Jacobians/{point}",
                     rtol=1e-12, atol=1e-13)
        assert_close(values["hessian"], np.asarray(values["hessian"]).T,
                     description=f"JAX Hessian/{point}", rtol=1e-12, atol=1e-12)
    assert jax_results["values"]["higher"]["fourth_polynomial"] == pytest.approx(90.0)
    assert np.all(np.isfinite(jax_results["values"]["higher"]["nested"]))


def test_require_eager_validation_and_unchecked_dynamic_predicates(jax_results):
    assert jax_results["require"] == {
        "python_true": None, "numerical_true": None,
        "python_false": "deliberate validation failure",
        "numerical_false": "deliberate validation failure",
    }
    values = jax_results["jax"]
    # This is deliberate: a runtime-false predicate inside JIT does NOT raise.
    assert values["dynamic_invalid"] == pytest.approx(4.0)
    assert values["static_false"] == "static failure"
    assert values["guarded_gradient"] == pytest.approx(4.0)
    assert values["guarded_hessian"] == pytest.approx(2.0)
    np.testing.assert_allclose(values["batched_guarded"], [1.0, 4.0, 9.0])
    for callback in ("debug_callback", "pure_callback", "io_callback"):
        assert callback not in values["require_jaxpr"]


@pytest.mark.parametrize("name", ["graph", "input", "gaussian", "ray", "bundle", "refractive", "aperture"])
def test_vmap_matches_scalar_numerical_evaluations(jax_results, name):
    expected = (np.array([values["values"] for values in jax_results["values"][name].values()])
                if name in ("graph", "input") else jax_results["values"]["optics"][name])
    assert_close(jax_results["jax"][f"batched_{name}"], expected,
                 description=f"JAX vmap/{name}", rtol=2e-10, atol=1e-11)
