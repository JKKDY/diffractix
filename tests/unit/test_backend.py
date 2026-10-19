"""Backend selection is process-wide, so configuration tests use fresh processes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_python(code, preference="auto"):
    env = dict(os.environ, DIFFRACTIX_BACKEND=preference)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("name", ["autograd", "jax"])
def test_import_is_lazy_and_configuration_can_be_overridden(name):
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"Optional numerical backend {name} is not installed")
    assert run_python(f'''
import sys
import diffractix
from diffractix import backend
assert backend.current_backend() is None
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
