"""Small process-isolation helpers; numerical backends are never switched in place."""
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def run_python(code, preference="auto", *, timeout=180):
    env = dict(os.environ, DIFFRACTIX_BACKEND=preference, JAX_ENABLE_X64="true")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=ROOT,
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, (
        f"Backend {preference!r} child failed ({result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout.strip()


@lru_cache(maxsize=None)
def numerical_results(name, group="representative"):
    """One plain Python child per backend/group, not a nested pytest invocation."""
    result = json.loads(run_python(f'''
import json
from tests.helpers.backend_cases import evaluate_cases
from diffractix.backend import current_backend
values = evaluate_cases({group!r})
assert current_backend() == {name!r}
print(json.dumps(values, allow_nan=False))
''', name))
    assert result["backend"] == name
    return result


def assert_close(actual, expected, *, description, rtol, atol):
    """Report the affected operation and discrepancy, without accepting NaNs."""
    import numpy as np

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), description
        for key in expected:
            assert_close(actual[key], expected[key], description=f"{description}/{key}",
                         rtol=rtol, atol=atol)
        return
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, description
    assert np.all(np.isfinite(actual)), f"{description}: non-finite actual {actual}"
    assert np.all(np.isfinite(expected)), f"{description}: non-finite expected {expected}"
    difference = np.abs(actual - expected)
    absolute = np.max(difference, initial=0.0)
    relative = np.max(difference / np.maximum(np.abs(expected), atol), initial=0.0)
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=False,
        err_msg=f"{description}: max absolute={absolute:.6g}, relative={relative:.6g}",
    )


def capture_problem(solver):
    """Exercise normal solver preparation without running an optimization backend."""
    from unittest.mock import patch
    from diffractix.solver import solver as solver_module

    captured = {}

    class Captured(Exception):
        pass

    def capture(problem, **kwargs):
        captured["problem"] = problem.unscale()
        raise Captured

    with patch.object(solver_module, "solve_scipy", capture):
        try:
            solver.solve()
        except Captured:
            pass
    return captured["problem"]


def example_before_solve(name):
    """Reuse the actual example's setup, stopping before optimization/reporting."""
    import ast
    from types import ModuleType

    path = ROOT / "examples" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefix = []
    for statement in tree.body:
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "solve" for node in ast.walk(statement)):
            break
        prefix.append(statement)
    else:
        raise AssertionError(f"No solve boundary in {path}")
    module = ModuleType(f"_backend_test_{name}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__
