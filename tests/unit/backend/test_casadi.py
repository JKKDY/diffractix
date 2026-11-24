"""CasADi tests use a fresh process, regardless of the test run's backend."""
import importlib.util

import pytest

from tests.helpers.backend import run_python


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="Optional CasADi backend is not installed",
)


def test_opt_in_and_lazy_selection():
    assert run_python("""
import sys
from diffractix import backend
assert backend.current_backend() is None
assert 'casadi' not in sys.modules
backend.configure('casadi')
assert 'casadi' not in sys.modules
assert backend.xp.add(1, 2) == 3
assert backend.current_backend() == 'casadi'
print('ok')
""", preference="invalid") == "ok"


def test_sx_transforms_share_trace_and_preserve_shapes():
    assert run_python("""
import numpy as np
from diffractix.backend import current_backend, grad, hessian, jacobian, jit, xp
seen = []

def cost(x):
    seen.append(x.shape)
    return xp.sum(x**3)

x = np.array([1., 2.])
assert current_backend() is None
assert jit(cost)(x) == 9
assert current_backend() == 'casadi'
np.testing.assert_allclose(grad(cost)(x), [3, 12])
np.testing.assert_allclose(hessian(cost)(x), [[6, 0], [0, 12]])
assert seen == [(2,)]
np.testing.assert_allclose(jacobian(lambda x: xp.stack((x, x*x)))(x),
                           [[[1, 0], [0, 1]], [[2, 0], [0, 4]]])
assert jit(lambda x, flag: x if flag else -x, static_argnums=1)(3., False) == -3
typed_static = jit(lambda x, value: x if type(value) is int else -x, static_argnums=1)
assert typed_static(3., 1) == 3
assert typed_static(3., True) == -3
assert jit(lambda x: x + 2j*x)(3.) == 3 + 6j
assert np.isnan(jit(lambda x: 0. / x)(0.))
assert jit(lambda x: 1. / x)(-0.) == -np.inf
for operation in (xp.minimum, xp.maximum):
    assert np.isnan(jit(lambda x: operation(xp.sqrt(x), 0.))(-1.))
assert jit(lambda x: xp.sum(x > 0))(x) == 2
assert jit(xp.sum)(np.array([120, 120], dtype=np.int8)) == 240
assert jit(lambda x: xp.dot(x, x))(np.empty(0)) == 0
z = np.array([complex(2., np.inf)])
assert jit(lambda x: x)(z)[0] == z[0]
print('ok')
""", preference="casadi") == "ok"


def test_tracing_errors_are_specific_and_static_validation_still_runs():
    assert run_python("""
import numpy as np
from diffractix.backend import jit, require, tracing_errors, xp
for function in (lambda x: x if x > 0 else -x, float, np.asarray, xp.floor):
    try:
        jit(function)(1.)
    except tracing_errors():
        pass
    else:
        raise AssertionError(function)
try:
    jit(lambda x: (require(False, 'static failure'), x)[1])(1.)
except ValueError as error:
    assert str(error) == 'static failure'
else:
    raise AssertionError('Static validation was skipped')
assert jit(lambda x: xp.where(x > 0, x, -x))(-2.) == 2
try:
    jit(lambda x: x)(2**53 + 1)
except tracing_errors() as error:
    assert 'double-precision range' in str(error)
else:
    raise AssertionError('An integer was silently rounded')
print('ok')
""", preference="casadi") == "ok"
