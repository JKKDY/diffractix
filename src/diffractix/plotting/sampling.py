"""Concrete plot data from boundary states and known homogeneous intervals."""
import numpy as np

from diffractix.input import GaussianBeam, Ray, RayBundle


def _concrete(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Plotting requires concrete numerical results outside differentiation or JIT."
        ) from error
    if array.dtype.kind == "c" and np.all(array.imag == 0):
        array = array.real
    if array.dtype.kind not in "biuf":
        raise ValueError(
            "Plotting requires concrete real numerical results outside differentiation or JIT."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Plotting requires finite numerical values.")
    return array


def sample(result):
    """Return (kind, boundary_z, boundary_values, independent_space_segments).

    Each segment is a pair of arrays (z, transverse). No segment spans a thin
    transformation or an unknown thick element. Coordinates retain metres.
    Gaussian waist locations are included explicitly in the sampling grid.
    Space intervals are identified by the existing element type_name metadata.
    """
    source = result.source
    if isinstance(source, GaussianBeam):
        kind = "beam"
    elif isinstance(source, RayBundle):
        kind = "bundle"
    elif isinstance(source, Ray):
        kind = "ray"
    else:
        raise TypeError(
            "Plotting supports numerical Ray, RayBundle and GaussianBeam states "
            f"and their subclasses; got {type(source).__name__}."
        )
    z = _concrete(result.z)
    if z.ndim != 1:
        raise ValueError("Plotting requires one scalar z position per state.")
    boundaries = np.stack([
        _concrete(state.w if kind == "beam" else state.x)
        for state in result.states
    ])
    expected_ndim = 2 if kind == "bundle" else 1
    if boundaries.ndim != expected_ndim:
        raise ValueError(f"Plotting requires {'1D ray arrays' if kind == 'bundle' else 'scalar state values'}.")

    indices = None
    if kind == "bundle":
        count = boundaries.shape[1]
        indices = np.linspace(0, count - 1, min(count, 41), dtype=int)
        boundaries = boundaries[:, indices]

    segments = []
    for index, info in enumerate(result.element_info):
        length = z[index + 1] - z[index]
        if info.type_name != "Space" or length == 0:
            continue
        state = result.states[index]
        distance = np.linspace(0, length, 129)
        if kind == "beam":
            waist = float(_concrete(state.z0))
            if min(0, length) < waist < max(0, length):
                distance = np.unique(np.append(distance, waist))
                if length < 0:
                    distance = distance[::-1]
            # w_at_z validates scalar q values; keep that existing interface.
            values = np.stack([_concrete(state.w_at_z(float(offset))) for offset in distance])
        else:
            local = distance[:, None] if kind == "bundle" else distance
            values = _concrete(state.x_at_z(local))
            if indices is not None:
                values = values[:, indices]
        segments.append((z[index] + distance, values))
    return kind, z, boundaries, segments
