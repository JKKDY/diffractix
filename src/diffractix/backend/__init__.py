"""Process-wide numerical backend selection for Diffractix.

The default preference ("auto") selects JAX when installed, otherwise
Autograd. Set "DIFFRACTIX_BACKEND" or call "configure(name)" *before* the
first numerical operation to select "auto", "jax" or "autograd".

Importing this module, "xp", or a differentiation wrapper does not initialize
the backend. The first "xp" attribute access or derivative/JIT construction
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

Do not bind "xp" attributes into module-level registries during import;
resolve those attributes lazily when preparing compiled numerical instructions.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from threading import RLock
from types import ModuleType
from typing import Any, Callable

__all__ = [
    "xp",
    "grad",
    "jacobian",
    "jacrev",
    "hessian",
    "jit",
    "require",
    "configure",
    "current_backend",
]

_CHOICES = frozenset({"auto", "jax", "autograd"})
_lock = RLock()
_requested = os.environ.get("DIFFRACTIX_BACKEND", "auto").strip().lower()
_active: _Backend | None = None


@dataclass(frozen=True)
class _Backend:
    name: str
    xp: ModuleType
    grad: Callable[..., Any]
    jacobian: Callable[..., Any]
    jacrev: Callable[..., Any]
    hessian: Callable[..., Any]
    jit: Callable[..., Any]
    require:Callable[[Any, str], None]

def configure(name: str) -> None:
    """Choose a backend before its first use; switching afterward is forbidden.

    The explicit choice overrides ``DIFFRACTIX_BACKEND`` in this process.
    Calling this function does *not* import either numerical backend.
    """
    global _requested

    if not isinstance(name, str):
        raise TypeError("Backend name must be 'auto', 'jax', or 'autograd'.")
    name = name.strip().lower()
    if name not in _CHOICES:
        raise ValueError(f"Unknown Diffractix backend {name!r}; choose auto, jax, or autograd.")

    with _lock:
        if _active is not None:
            raise RuntimeError(
                f"Diffractix numerical backend is already initialized as "
                f"{_active.name!r}; restart Python to change backend configuration."
            )
        _requested = name


def current_backend() -> str | None:
    """Return the active backend name, or None before first numerical use."""
    with _lock:
        return None if _active is None else _active.name


def _identity_jit(fun: Callable[..., Any] | None = None, **_options: Any) -> Any:
    """Autograd's no-op equivalent of JAX's JIT decorator/function."""
    return fun


def _load_autograd() -> _Backend:
    try:
        import autograd
        import autograd.numpy as anp
    except ImportError as exc:
        raise ImportError(
            "Diffractix requires Autograd for this backend. "
            "Install it with 'pip install autograd' (or install diffractix)."
        ) from exc

    def _eager_require(condition: Any, message: str) -> None:
        """Require a concrete scalar condition to be true."""
        if not bool(condition):
            raise ValueError(message)

    return _Backend(
        name="autograd",
        xp=anp,
        grad=autograd.grad,
        jacobian=autograd.jacobian,
        jacrev=autograd.jacobian,
        hessian=autograd.hessian,
        jit=_identity_jit,
        require=_eager_require,
    )


def _load_jax() -> _Backend:
    try:
        import jax
    except ImportError as exc:
        raise ImportError(
            "Diffractix cannot import JAX. Install it with "
            "'pip install \"diffractix[jax]\"', or select Autograd instead."
        ) from exc

    # use 64 bit 
    if "JAX_ENABLE_X64" not in os.environ:
        jax.config.update("jax_enable_x64", True)

    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "Diffractix found JAX but could not import jax.numpy; "
            "check the JAX/JAXLIB installation."
        ) from exc

    def _jax_require(condition, message: str) -> None:
        try:
            valid = bool(condition)
        except jax.errors.TracerBoolConversionError:
            return

        if not valid:
            raise ValueError(message)

        return _Backend(
            name="jax",
            xp=jnp,
            grad=jax.grad,
            jacobian=jax.jacfwd,
            jacrev=jax.jacrev,
            hessian=jax.hessian,
            jit=jax.jit,
            require=_jax_require,
        )


def _initialize(preference: str) -> _Backend:
    if preference not in _CHOICES:
        raise ValueError(
            f"Invalid DIFFRACTIX_BACKEND={preference!r}; use auto, jax, or autograd."
        )
    if preference == "autograd":
        return _load_autograd()
    if preference == "jax":
        return _load_jax()
    if preference == "auto":
        return _load_jax() if importlib.util.find_spec("jax") is not None else _load_autograd()


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
    """Construct a gradient using the selected backend."""
    return _get_backend().grad(fun, *args, **kwargs)


def jacobian(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct a Jacobian (Autograd jacobian / JAX jacfwd)."""
    return _get_backend().jacobian(fun, *args, **kwargs)


def jacrev(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct a reverse-mode Jacobian where supported."""
    return _get_backend().jacrev(fun, *args, **kwargs)


def hessian(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Construct a Hessian using the selected backend."""
    return _get_backend().hessian(fun, *args, **kwargs)


def jit(fun: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """JAX JIT, or an identity function/decorator under Autograd.

    Supports both ``jit(fun)`` and ``@jit(static_argnums=...)``. The latter
    accepts and ignores JAX-only compilation options under Autograd.
    """
    if fun is None:
        # Works with older JAX versions that do not support @jax.jit(options).
        return lambda f: _get_backend().jit(f, **kwargs)
    return _get_backend().jit(fun, **kwargs)


def require(condition: Any, message: str) -> None:
    """Require a scalar numerical condition to be true.

    Under eager execution, an invalid condition raises ValueError immediately.

    Under JAX JIT, the condition is represented with JAX control flow and the
    error is raised from the invalid runtime branch.
    """
    _get_backend().require(condition, message)