"""Lazy optimizer entry points.

Libraries are checked only when their selected optimizer is called.
This keeps importing :mod:`diffractix.solver` independent of IPOPT and NLopt.
"""

from importlib import import_module


def _require_package(package: str, extra: str | None, hint: str = "") -> None:
    try:
        import_module(package)
    except ImportError as error:
        install = f"diffractix[{extra}]" if extra else "diffractix"
        label = extra.upper() if extra else "NLopt"
        raise ImportError(f"The {label} optimizer requires {package!r}. Install {install}.{hint}") from error


def solve_scipy(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the SciPy optimizer."""
    options = {} if options is None else dict(options)
    _require_package("scipy", "scipy")
    from .scipy import solve_scipy as solve

    return solve(problem, method=method, options=options)


def solve_ipopt(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the optional IPOPT optimizer."""
    options = {} if options is None else dict(options)
    _require_package(
        "cyipopt", "ipopt",
        " pip builds cyipopt from source and needs Ipopt installed; the simplest"
        " route is: conda install -c conda-forge cyipopt",
    )
    from .ipopt import solve_ipopt as solve

    return solve(problem, method=method, options=options)


def solve_nlopt(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the NLopt optimizer."""
    options = {} if options is None else dict(options)
    _require_package("nlopt", None)
    from .nlopt import solve_nlopt as solve

    return solve(problem, method=method, options=options)


__all__ = ["solve_scipy", "solve_ipopt", "solve_nlopt"]
