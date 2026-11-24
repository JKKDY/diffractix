"""Process-wide numerical backend selection for Diffractix.

The default preference (`auto`) selects JAX when installed, otherwise
Autograd. Set `DIFFRACTIX_BACKEND` or call `configure(name)` before the
first numerical operation to select `auto`, `jax`, `autograd` or `casadi`.

`xp` provides the selected backend's differentiable NumPy interface:
`jax.numpy`, `autograd.numpy` or the CasADi NumPy interface.

Importing this module, `xp`, or a differentiation wrapper does not initialize
the backend. The first `xp` attribute access or derivative/JIT construction
resolves and permanently freezes the backend for the current Python process.

Numerical modules should use::

    import numpy as np
    from diffractix.backend import (
        xp,
        grad,
        jacobian,
        hessian,
        jit,
        require,
    )

Do not bind `xp` attributes into module-level registries during import;
resolve those attributes lazily when preparing compiled numerical instructions.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

__all__ = [
    "xp",
    "grad",
    "jacobian",
    "hessian",
    "jit",
    "require",
    "tracing_errors",
    "configure",
    "current_backend",
]

_CHOICES = frozenset({"auto", "jax", "autograd", "casadi"})
_lock = RLock()
_requested = os.environ.get("DIFFRACTIX_BACKEND", "auto").strip().lower()
_active: _Backend | None = None


@dataclass(frozen=True)
class _Backend:
    name: str
    xp: Any
    grad: Callable[..., Any]
    jacobian: Callable[..., Any]
    hessian: Callable[..., Any]
    jit: Callable[..., Any]
    require:Callable[[Any, str], None]
    # Errors raised when a function uses a traced value where a concrete one is needed.
    tracing_errors: tuple[type[BaseException], ...] = ()

def configure(name: str) -> None:
    """Choose the numerical backend. Must be run before first use. 

    Parameters
    ----------
    name : str
        ``"auto"``, ``"jax"``, ``"autograd"`` or ``"casadi"``. Overrides
        ``DIFFRACTIX_BACKEND`` for this process. Does not import a backend.

    Raises
    ------
    TypeError
        If `name` is not a string.
    ValueError
        If `name` is not one of the supported choices.
    RuntimeError
        If a backend is already initialized; restart Python to change it.
    """
    global _requested

    if not isinstance(name, str):
        raise TypeError("Backend name must be 'auto', 'jax', 'autograd', or 'casadi'.")
    name = name.strip().lower()
    if name not in _CHOICES:
        raise ValueError(f"Unknown Diffractix backend {name!r}; choose auto, jax, autograd, or casadi.")

    with _lock:
        if _active is not None:
            raise RuntimeError(
                f"Diffractix numerical backend is already initialized as "
                f"{_active.name!r}; restart Python to change backend configuration."
            )
        _requested = name


def current_backend() -> str | None:
    """Return the active backend's name, or ``None`` before the first numerical use.

    Returns
    -------
    str or None
        The backend's name, or ``None`` if it has not yet been initialized.
    """
    with _lock:
        return None if _active is None else _active.name


def _initialize(preference: str) -> _Backend:
    if preference not in _CHOICES:
        raise ValueError(
            f"Invalid DIFFRACTIX_BACKEND={preference!r}; use auto, jax, autograd, or casadi."
        )
    if preference == "auto":
        preference = "jax" if importlib.util.find_spec("jax") is not None else "autograd"

    if preference == "autograd":
        from ._autograd import _load_autograd
        return _load_autograd()
    if preference == "jax":
        from ._jax import _load_jax
        return _load_jax()
    if preference == "casadi":
        from ._casadi import _load_casadi
        return _load_casadi()


def _get_backend() -> _Backend:
    global _active
    with _lock:
        if _active is None:
            _active = _initialize(_requested)
        return _active


class _LazyXP:
    """Stable proxy that resolves/caches numerical attributes on first access."""

    def __getattr__(self, name: str) -> Any:
        value = getattr(_get_backend().xp, name)
        # The backend cannot change now; avoid repeated proxy dispatch.
        self.__dict__[name] = value
        return value

    def __repr__(self) -> str:
        return f"<diffractix.backend.xp: {current_backend() or 'uninitialized'}>"


xp = _LazyXP()


def grad(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct the gradient of `fun`, using the selected backend.

    Parameters
    ----------
    fun : callable
        Function to differentiate.
    *args, **kwargs
        Passed on to the backend's own gradient constructor.

    Returns
    -------
    callable
        The gradient function.
    """
    return _get_backend().grad(fun, *args, **kwargs)


def jacobian(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct the Jacobian of `fun` using the selected backend.

    Parameters
    ----------
    fun : callable
        Function to differentiate.
    *args, **kwargs
        Passed on to the backend's own Jacobian constructor.

    Returns
    -------
    callable
        The Jacobian function.
    """
    return _get_backend().jacobian(fun, *args, **kwargs)


def hessian(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct the Hessian of `fun`, using the selected backend.

    Parameters
    ----------
    fun : callable
        Function to differentiate.
    *args, **kwargs
        Passed on to the backend's own Hessian constructor.

    Returns
    -------
    callable
        The Hessian function.
    """
    return _get_backend().hessian(fun, *args, **kwargs)


def jit(fun: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Compile `fun` under JAX or CasADi, or return it unchanged under Autograd.

    Parameters
    ----------
    fun : callable, optional
        Function to compile. Omit it to use `jit` as a decorator factory, as in
        ``@jit(static_argnums=...)``.
    **kwargs
        JAX compilation options, or ``static_argnums`` under CasADi.
        Autograd accepts and ignores these options.

    Returns
    -------
    callable
        The compiled function, or a decorator if `fun` was omitted.
    """
    if fun is None:
        # Works with older JAX versions that do not support @jax.jit(options).
        return lambda f: _get_backend().jit(f, **kwargs)
    return _get_backend().jit(fun, **kwargs)


def tracing_errors() -> tuple[type[BaseException], ...]:
    """Exception types raised when JIT tracing cannot give a concrete value
    where one is needed.

    Returns
    -------
    tuple of type
        Empty under Autograd, which never traces abstractly.
    """
    return _get_backend().tracing_errors


def require(condition: Any, message: str) -> None:
    """Require a scalar numerical condition to be true.

    Parameters
    ----------
    condition : Any
        The condition.
    message : str
        Error message if `condition` is false.

    Raises
    ------
    ValueError
        If `condition` is false and can be checked; see Notes.

    Notes
    -----
    During JAX JIT or CasADi SX tracing, the inputs are symbolic placeholders.
    Predicates that depend on them are skipped. Concrete predicates are checked
    during compilation. Eager calls check every condition.
    """
    _get_backend().require(condition, message)
