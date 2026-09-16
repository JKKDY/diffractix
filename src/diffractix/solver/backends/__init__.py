"""Lazy optimization-backend entry points.

Optional libraries are checked only when their selected backend is called.
This keeps importing :mod:`diffractix.solver` independent of IPOPT and NLopt.
"""

from importlib import import_module


def _require_package(package: str, extra: str) -> None:
    try:
        import_module(package)
    except ImportError as error:
        raise ImportError(
            f"The {extra.upper()} solver backend requires optional package "
            f"{package!r}. Install diffractix[{extra}]."
        ) from error


def solve_scipy(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the SciPy backend."""
    _require_package("scipy", "scipy")
    from .scipy import solve_scipy as solve

    return solve(problem, method=method, options=options)


def solve_ipopt(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the optional IPOPT backend."""
    _require_package("cyipopt", "ipopt")
    from .ipopt import solve_ipopt as solve

    return solve(problem, method=method, options=options)


def solve_nlopt(problem, method: str | None = None, options: dict | None = None):
    """Dispatch to the optional NLopt backend."""
    _require_package("nlopt", "nlopt")
    from .nlopt import solve_nlopt as solve

    return solve(problem, method=method, options=options)


__all__ = ["solve_scipy", "solve_ipopt", "solve_nlopt"]
