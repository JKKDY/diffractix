"""2D paraxial plotting; Matplotlib loads only when plot() is called."""

__all__ = ["plot"]


def plot(result, ax=None, *, show_elements=True, show_axis=True):
    """Plot a concrete 2D trace of a result.

    Equivalent to :meth:`SimulationResult.plot`; only known homogeneous Spaces
    are sampled between boundaries; see Notes.

    Parameters
    ----------
    result : SimulationResult
        The result to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to ``None``: a new figure and axes are created.
    show_elements : bool, optional
        Whether to draw and label the elements. Defaults to ``True``.
    show_axis : bool, optional
        Whether to draw the optical axis. Defaults to ``True``.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Neither is shown or saved; call ``fig.savefig(...)`` or ``plt.show()``
        yourself.

    Raises
    ------
    ImportError
        If Matplotlib cannot be imported.

    Notes
    -----
    ``z`` is shown in metres, transverse quantities in millimetres. Supports
    `Ray`, `RayBundle` and `GaussianBeam` results; a bundle shows up to 41 rays. A
    thick element such as `GRIN` retains only its boundary states; the interior
    is not sampled. General probing at an arbitrary z is not supported.
    """
    try:
        from .matplotlib import plot as render
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "matplotlib":
            raise ImportError(
                "Plotting requires Matplotlib, which is installed with Diffractix. "
                "Repair the installation with python -m pip install matplotlib."
            ) from error
        raise
    return render(result, ax=ax, show_elements=show_elements, show_axis=show_axis)
