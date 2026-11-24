"""JAX numerical backend."""

import os

from . import _Backend


def _load_jax() -> _Backend:
    try:
        import jax
    except ImportError as exc:
        raise ImportError(
            "Diffractix cannot import JAX. Install it with "
            "'pip install \"diffractix[jax]\"', or select another backend."
        ) from exc

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
        jacobian=jax.jacrev,
        hessian=jax.hessian,
        jit=jax.jit,
        require=_jax_require,
        tracing_errors=(
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
            jax.errors.TracerIntegerConversionError,
        ),
    )
