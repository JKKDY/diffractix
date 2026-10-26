"""Importing and simulating must not depend on the optional renderer."""
from tests.helpers.backend import run_python


def test_matplotlib_is_lazy_and_missing_dependency_has_install_instructions():
    assert run_python('''
import importlib.abc
import sys
class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("Matplotlib intentionally unavailable", name="matplotlib")
sys.meta_path.insert(0, NoMatplotlib())
import diffractix
from diffractix.plotting import plot
assert not any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules)
result = diffractix.System(input=diffractix.Ray(x=0.0, theta=0.0)).build().run()
assert result.source.x == 0.0
for render in (result.plot, lambda: plot(result)):
    try:
        render()
    except ImportError as error:
        assert "Matplotlib" in str(error)
        assert "pip install" in str(error) and "diffractix[plot]" in str(error)
    else:
        raise AssertionError("Missing optional dependency must fail clearly")
print("ok")
''', "autograd") == "ok"
