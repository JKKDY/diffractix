"""Autograd-specific backend contracts, isolated from the pytest backend."""
from tests.helpers.backend import run_python


def test_autograd_identity_jit_derivatives_and_concrete_validation():
    assert run_python('''
import numpy as np
import pytest
from diffractix.backend import current_backend, grad, hessian, jacobian, jit, require, xp
function = lambda theta, power=3: xp.stack((theta[0]**power, theta[0]*theta[1]))
assert jit(function) is function
assert jit(static_argnums=1)(function) is function
assert current_backend() == 'autograd'
point = xp.array([2., 3.])
np.testing.assert_allclose(jacobian(function)(point), [[12., 0.], [3., 2.]])
cost = lambda theta: xp.real((theta[0]+1j*theta[1])**2 * xp.conj((theta[0]+1j*theta[1])**2))
np.testing.assert_allclose(grad(cost)(xp.array([0.6, -0.2])), [0.96, -0.32])
np.testing.assert_allclose(hessian(cost)(xp.array([0.6, -0.2])), [[4.48, -0.96], [-0.96, 1.92]])
for condition in (True, xp.array(True)):
    assert require(condition, 'deliberate failure') is None
for condition in (False, xp.array(False)):
    with pytest.raises(ValueError, match='deliberate failure'):
        require(condition, 'deliberate failure')
print('ok')
''', 'autograd') == 'ok'
