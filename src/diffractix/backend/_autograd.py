"""Autograd numerical backend."""

from typing import Any, Callable

from . import _Backend


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
        hessian=autograd.hessian,
        jit=_identity_jit,
        require=_eager_require,
    )
