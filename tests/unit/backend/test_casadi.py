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


def test_solver_reuses_one_optics_trace_and_preserves_callbacks():
    assert run_python("""
from unittest.mock import patch

import numpy as np
from diffractix import GaussianBeam, Plane, Solver, Space, System, ThinLens
from diffractix.backend import _solver_callback_compiler
from diffractix.backend._casadi_array import SXArray
from diffractix.simulation import Simulation
from diffractix.solver import solver as solver_module

traced_inputs = []
traced_quantities = []
run = Simulation.run
resolve = solver_module.resolve_symbol

def counted_run(self, theta=None):
    if isinstance(theta, SXArray):
        traced_inputs.append(theta)
    return run(self, theta)

def counted_resolve(key, result):
    if isinstance(result.theta, SXArray):
        traced_quantities.append(key)
    return resolve(key, result)

for count in (2, 4, 8):
    spaces = [Space(d=0.06).variable(d=(0.01, 0.4)) for _ in range(count)]
    lenses = [ThinLens(f=(0.08, 0.12)[i % 2]).variable(f=(0.02, 0.4)) for i in range(count)]
    image = Plane()
    parts = [element for pair in zip(spaces, lenses) for element in pair]
    sim = System(*parts, Space(d=0.15), image,
                 source=GaussianBeam.from_waist(0.0005, 1.064e-6)).build()
    symbolic = sim.symbolic
    assert not sim._execution_cache

    def make_solver(goal, focus):
        solver = Solver(sim)
        solver.target(value=symbolic[image].after.w, target=goal, scale=0.00012)
        if focus:
            solver.target(value=symbolic[image].after.w0, target=goal, scale=0.00012)
        for lens in lenses:
            solver.require(symbolic[lens].after.w <= 0.002)
        return solver

    theta = sim.initial_values.values
    lo = np.array([p.lower_bound for p in sim.parameter_data])
    hi = np.array([p.upper_bound for p in sim.parameter_data])
    with patch.object(Simulation, 'run', counted_run), patch.object(solver_module, 'resolve_symbol', counted_resolve):
        first = make_solver(0.00012, False)
        start = len(traced_inputs)
        first.solve()
        assert len(traced_inputs) == start + 1
        assert len(sim._execution_cache) == 1
        trace, = sim._execution_cache.values()
        assert traced_inputs[-1] is trace.theta
        assert len(trace._quantities) == count + 1
        start = len(traced_inputs)
        quantities_start = len(traced_quantities)
        second = make_solver(0.00011998, True)
        new_guess = theta * 1.01
        lo[0] = hi[0] = new_guess[0]
        second.solve(x0=new_guess, x_lower=lo, x_upper=hi)
        assert len(traced_inputs) == start
        assert len(traced_quantities) == quantities_start + 1

    # Bypass only the solver's cache selection. Both sides still use the CasADi
    # backend and identical solver settings, so callback parity can be exact.
    def uncached_compiler(cache, initial_values, run, resolve, **options):
        options['has_result_symbols'] = False
        return _solver_callback_compiler(cache, initial_values, run, resolve, **options)

    with patch.object(solver_module, '_solver_callback_compiler', uncached_compiler):
        reference = make_solver(0.00011998, True)
        reference.solve(x0=new_guess, x_lower=lo, x_upper=hi)
    sample = theta * 1.03
    for name in ('objective', 'gradient', 'constraints', 'jacobian', 'objective_hessian'):
        actual = getattr(second._cache.callbacks, name)(sample)
        expected = getattr(reference._cache.callbacks, name)(sample)
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes(), name
print('ok')
""", preference="casadi") == "ok"
