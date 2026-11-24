"""Matplotlib is required only by rendering methods."""
import os

from tests.helpers.backend import run_python


def test_import_and_solve_without_matplotlib():
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
lens = diffractix.ThinLens(f=0.2).variable(f=(0.05, 0.5))
system = diffractix.System(lens, diffractix.Space(d=0.1),
                          source=diffractix.Ray(x=0.01, theta=0.0))
solver = diffractix.Solver(system).target(lambda result: result.final.x / 0.01)
solution = solver.solve()
assert solution.success and abs(solution.run().final.x) < 1e-5
assert not any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules)
result = solution.run()
beam = diffractix.GaussianBeam.from_waist(0.001, 1e-6, n=1.)
for render in (result.plot, lambda: plot(result), beam.plot):
    try:
        render()
    except ImportError as error:
        assert "Matplotlib" in str(error)
        assert "pip install matplotlib" in str(error)
        assert "diffractix[plot]" in str(error)
    else:
        raise AssertionError("A missing plotting dependency must fail clearly")
print("ok")
''', os.environ.get("DIFFRACTIX_BACKEND", "auto")) == "ok"
