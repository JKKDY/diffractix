"""Exercise the installed wheel; run with its clean venv's python -I."""
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diffractix
from diffractix import Plane, Ray, Solver, System, ThinLens
from diffractix.backend import current_backend


def main():
    location = Path(diffractix.__file__).resolve()
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Imported {location} outside the clean environment {sys.prefix}.")
    for name in diffractix.__all__:
        if not hasattr(diffractix, name):
            raise RuntimeError(f"Missing public export: {name}")

    lens = ThinLens(f=0.2)
    lens.f.variable(bounds=(0.05, 0.5))
    target = Plane(label="Image plane")
    system = System(input=Ray(x=0.01, theta=0.0))
    system.add(lens, z=0.1).add(target, z=0.2)
    simulation = system.build()
    np.testing.assert_allclose(simulation.run().at(target).x, 0.005, atol=1e-12)
    solution = Solver(simulation).target(
        value=lambda ctx: ctx.at(target).x, target=0.0, scale=0.01,
    ).solve(method="SLSQP", options={"ftol": 1e-12})
    if not solution.success or not solution.feasible:
        raise RuntimeError(f"Inverse-design smoke failed: {solution.message}")
    np.testing.assert_allclose(solution[lens.f], 0.1, atol=1e-6)
    np.testing.assert_allclose(solution.run().at(target).x, 0.0, atol=1e-6)
    np.testing.assert_allclose(lens.f.value, 0.2)
    fig, ax = solution.run().plot()
    if ax.figure is not fig:
        raise RuntimeError("Plot did not return its Figure and Axes.")
    plt.close(fig)
    print(f"Wheel smoke passed: diffractix {version('diffractix')} at {location}; backend={current_backend()}")


if __name__ == "__main__":
    main()
