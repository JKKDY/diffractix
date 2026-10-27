"""Run every standalone example with Agg and verify its numerical backend."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import os
import runpy
import sys
from diffractix.backend import current_backend, xp

xp.asarray(0.0)
expected = os.environ["DIFFRACTIX_BACKEND"]
if current_backend() != expected:
    raise RuntimeError(f"Expected {expected}, got {current_backend()}")
namespace = runpy.run_path(sys.argv[1], run_name="__main__")
if current_backend() != expected:
    raise RuntimeError("The example changed the selected numerical backend.")
solution = namespace.get("solution")
if solution is not None and (not solution.success or not solution.feasible):
    raise RuntimeError(f"Example optimization failed or is infeasible: {solution.message}")
import matplotlib.pyplot as plt
plt.close("all")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("autograd", "jax"), default="autograd")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds allowed per example")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # Underscore-prefixed modules are reserved for helpers, not programs.
    examples = sorted(path for path in (ROOT / "examples").glob("*.py") if not path.name.startswith("_"))
    if not examples:
        raise RuntimeError("No standalone examples found.")
    env = dict(os.environ, DIFFRACTIX_BACKEND=args.backend, MPLBACKEND="Agg", PYTHONUTF8="1")
    for path in examples:
        print(f"\n=== {path.name} [{args.backend}] ===", flush=True)
        subprocess.run([sys.executable, "-c", CHILD, str(path)], cwd=ROOT,
                       env=env, check=True, timeout=args.timeout)
    print(f"\nAll {len(examples)} examples passed with {args.backend}.", flush=True)


if __name__ == "__main__":
    main()
