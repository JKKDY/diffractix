"""Backend selection is process-wide, so configuration tests use fresh processes."""
import importlib.util
import pytest
from tests.helpers.backend import JAX_AVAILABLE, run_python


@pytest.mark.parametrize("name", ["autograd", "jax"])
def test_import_is_lazy_and_configuration_can_be_overridden(name):
    if name == "jax" and not JAX_AVAILABLE:
        pytest.skip(f"Optional numerical backend {name} is not installed")
    assert run_python(f'''
import sys
import diffractix
from diffractix import backend
from diffractix.backend import xp, grad, jacobian, jacrev, hessian, jit, require
assert backend.current_backend() is None
assert 'uninitialized' in repr(xp)
assert 'jax' not in sys.modules
assert 'autograd' not in sys.modules
backend.configure({name!r})
assert backend.current_backend() is None
assert backend.xp.add(2., 3.) == 5.
assert backend.current_backend() == {name!r}
try:
    backend.configure({name!r})
except RuntimeError:
    pass
else:
    raise AssertionError('Numerical initialization must freeze configuration')
print('ok')
''', preference="invalid") == "ok"


def test_auto_selection_and_environment_preference():
    expected = "jax" if importlib.util.find_spec("jax") else "autograd"
    assert run_python('''
from diffractix.backend import xp, current_backend
xp.add(1., 2.)
print(current_backend())
''') == expected
    assert run_python('''
from diffractix.backend import xp, current_backend
xp.add(1., 2.)
print(current_backend())
''', "autograd") == "autograd"


def test_invalid_configuration_does_not_initialize_backend():
    assert run_python('''
from diffractix import backend
for value, error in [(None, TypeError), ('unknown', ValueError)]:
    try:
        backend.configure(value)
    except error:
        pass
    else:
        raise AssertionError(value)
assert backend.current_backend() is None
print('ok')
''') == "ok"


@pytest.mark.parametrize("name", ["autograd", "jax", "auto"])
def test_each_environment_preference_and_programmatic_choice(name):
    if name == "jax" and not JAX_AVAILABLE:
        pytest.skip("Optional JAX numerical backend is not installed")
    expected = "jax" if name == "jax" or (name == "auto" and JAX_AVAILABLE) else "autograd"
    for setup, preference in (("", name), (f"backend.configure({name!r})", "invalid")):
        assert run_python(f'''
from diffractix import backend
{setup}
assert backend.current_backend() is None
assert backend.xp.add(1., 2.) == 3.
print(backend.current_backend())
''', preference) == expected


def test_invalid_environment_fails_only_on_numerical_use():
    assert run_python('''
from diffractix import backend
assert backend.current_backend() is None
try:
    backend.xp.add(1., 2.)
except ValueError as error:
    assert 'DIFFRACTIX_BACKEND' in str(error)
else:
    raise AssertionError('Invalid preference must not select a backend')
assert backend.current_backend() is None
print('ok')
''', "unknown") == "ok"


def test_parallel_first_use_has_one_consistent_backend():
    expected = "jax" if JAX_AVAILABLE else "autograd"
    assert run_python(f'''
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from diffractix import backend
assert backend.current_backend() is None
barrier = Barrier(8)
def first_use(index):
    barrier.wait()
    add = backend.xp.add
    return backend.current_backend(), id(add), float(add(index, 1.))
with ThreadPoolExecutor(max_workers=8) as workers:
    results = list(workers.map(first_use, range(8)))
assert {{result[0] for result in results}} == {{{expected!r}}}
assert len({{result[1] for result in results}}) == 1
assert [result[2] for result in results] == list(range(1, 9))
try:
    backend.configure('autograd')
except RuntimeError:
    pass
else:
    raise AssertionError('Parallel initialization must freeze configuration')
print('ok')
''') == "ok"


def test_auto_falls_back_when_jax_is_not_available():
    assert run_python('''
import importlib.util
original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: None if name == 'jax' else original(name, *a, **k)
from diffractix.backend import xp, current_backend
xp.add(1., 2.)
assert current_backend() == 'autograd'
print('ok')
''') == "ok"


def test_explicit_missing_jax_fails_without_silent_fallback():
    assert run_python('''
import builtins
from diffractix import backend
original = builtins.__import__
def importing(name, *a, **k):
    if name == 'jax':
        raise ImportError('Simulated optional dependency absence')
    return original(name, *a, **k)
builtins.__import__ = importing
backend.configure('jax')
try:
    backend.xp.add(1., 2.)
except ImportError as error:
    assert 'JAX' in str(error)
else:
    raise AssertionError('An explicit backend must not fall back')
assert backend.current_backend() is None
print('ok')
''') == "ok"
