"""Validate an actual default solve with and without HiGHS available."""

import importlib.util
import json
import os
import subprocess
import sys
from textwrap import dedent

import pytest


def test_affine_gap_reduction_and_default_solve_without_scipy():
    pytest.importorskip("nlopt")
    code = dedent("""
        import json
        import sys
        from unittest.mock import patch

        blocked = sys.argv[1] == "numpy"
        if blocked:
            sys.modules["scipy"] = None

        from diffractix import Parameter, Plane, Ray, Solver, System
        from diffractix.solver import lp

        assert "scipy.optimize" not in sys.modules
        z1 = Parameter(0.04).variable(bounds=(0.02, 0.10))
        z2 = Parameter(0.08).variable(bounds=(0.04, 0.12))
        system = System(source=Ray(x=0, theta=0))
        system.add(Plane(), z=z1).add(Plane(), z=z2)
        solver = Solver(system.build()).target(z1 - 0.06, z2 - 0.07)
        solver.require(z2 >= 0.05, label="minimum position")
        solver.require(2*z2 - z1 >= 0.05, label="implied by gap and minimum position")

        with patch.object(lp, "_solve_numpy", wraps=lp._solve_numpy) as fallback:
            solution = solver.solve()
            assert (fallback.call_count > 0) == blocked
        assert solution.success and solution.feasible, solution.message
        assert abs(solution.value(z1) - 0.06) < 1e-7
        assert abs(solution.value(z2) - 0.07) < 1e-7
        retained = [descriptor.label for descriptor in solver._cache.active_descriptors]
        assert any("Gap before" in (label or "") for label in retained)
        assert "implied by gap and minimum position" not in retained
        if blocked:
            assert sys.modules["scipy"] is None
            assert not any(name.startswith("scipy.") for name in sys.modules)
        print(json.dumps(retained))
    """)

    def retained(path):
        result = subprocess.run(
            [sys.executable, "-c", code, path], capture_output=True, text=True,
            env={**os.environ, "DIFFRACTIX_BACKEND": "autograd"}, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    numpy_retained = retained("numpy")
    if importlib.util.find_spec("scipy") is not None:
        assert retained("highs") == numpy_retained
