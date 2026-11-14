"""JAX-specific transformations; no reconfiguration of the pytest process."""
import pytest
from tests.helpers.backend import JAX_AVAILABLE, run_python

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="Optional JAX numerical backend is not installed")


def test_concrete_backend_scalar_arrays_can_be_captured_as_literal_constants():
    assert run_python('''
import numpy as np
from diffractix.backend import xp
from diffractix.graph import Parameter, compile_graph
p = Parameter(1.0).variable()
for scalar in (2, 2.0, 2.0 + 1j):
    graph = compile_graph((p + xp.array(scalar),))
    assert graph.parameter_indices == {id(p): 0}
    for theta in (1.0, 4.0):
        np.testing.assert_allclose(graph.evaluate(xp.array([theta])), [theta + scalar])
print('ok')
''', 'jax') == 'ok'


def test_backend_vector_arrays_are_captured_as_owned_literal_constants():
    assert run_python('''
import numpy as np
from diffractix.backend import xp
from diffractix.graph import Parameter
p = Parameter(1.0).variable()
for value in (xp.array([2.0]), xp.array([2.0, 3.0])):
    literal = (p + value).right
    np.testing.assert_array_equal(literal.value, np.asarray(value))
    assert not literal.value.flags.writeable
print('ok')
''', 'jax') == 'ok'


def test_active_jax_tracers_cannot_be_captured_as_literal_constants():
    assert run_python('''
import pytest
from diffractix.backend import jit, xp
from diffractix.graph import Parameter
p = Parameter(1.0).variable()
for value in (xp.array(2.0), xp.array([2.0, 3.0])):
    with pytest.raises(TypeError, match='concrete'):
        jit(lambda value: p + value)(value)
print('ok')
''', 'jax') == 'ok'


@pytest.mark.parametrize("kind", ["objective", "constraint"])
def test_solver_requested_jit_propagates_dynamic_control_flow_errors(kind):
    code = '''
import pytest
from jax.errors import TracerBoolConversionError
from diffractix import Constraint, Parameter, Ray, Solver, System
from diffractix.backend import xp

parameter = Parameter(0.5).variable()
solver = Solver(System(optical_input=Ray(x=parameter, theta=0, n=1)))

def dynamic(result):
    return result.final.x ** 2 if result.final.x > 0 else 5.0

if KIND == 'objective':
    solver.target(dynamic)
else:
    solver.require(Constraint(dynamic))
cache = solver._build_cache(True, solver.simulation.initial_values.values)
callback = cache.callbacks.objective if KIND == 'objective' else cache.callbacks.constraints
with pytest.raises(TracerBoolConversionError):
    callback(xp.array([0.5]))
print('ok')
'''
    assert run_python(code.replace("KIND", repr(kind)), "jax") == "ok"




def test_jax_transformations_precision_and_validation_boundary():
    assert run_python('''
import jax
import numpy as np
import pytest
from diffractix.backend import current_backend, grad, hessian, jacobian, jit, require, xp
assert str(xp.array(1.).dtype) == 'float64'
assert current_backend() == 'jax'
@jit(static_argnums=1)
def function(theta, power):
    return xp.stack((theta[0]**power, theta[0]*theta[1]))
value = lambda theta: function(theta, 3)
point = xp.array([2., 3.])
np.testing.assert_allclose(value(point), [8., 6.])
np.testing.assert_allclose(jacobian(value)(point), [[12., 0.], [3., 2.]])
points = xp.array([[2., 3.], [1., 4.]])
np.testing.assert_allclose(jit(jax.vmap(value))(points), [[8., 6.], [1., 4.]])
def guarded(x):
    require(x > 0, 'must be positive')
    return x*x
with pytest.raises(ValueError, match='must be positive'):
    guarded(-2.)
# Backend contract: traced predicates are intentionally not checked at runtime.
assert jit(guarded)(-2.) == 4
assert jit(grad(guarded))(2.) == 4
assert jit(hessian(guarded))(2.) == 2
np.testing.assert_allclose(jit(jax.vmap(guarded))(xp.array([1., -2., 3.])), [1., 4., 9.])
with pytest.raises(ValueError, match='static failure'):
    jit(lambda x: (require(False, 'static failure'), x)[1])(1.)
for callback in ('debug_callback', 'pure_callback', 'io_callback'):
    assert callback not in str(jax.make_jaxpr(guarded)(2.))
print('ok')
''', 'jax') == 'ok'
