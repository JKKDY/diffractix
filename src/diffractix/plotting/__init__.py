"""Optional 2D paraxial plotting; Matplotlib loads only when plot() is called."""

__all__ = ["plot"]


def plot(result, ax=None, *, show_elements=True, show_axis=True):
    """Plot an evaluated result, returning (figure, axes) without calling show().

    Lengths are interpreted as metres: z is displayed in m, transverse values
    in mm. Only known Space intervals have continuous trajectories; other
    elements show boundary values. Bundles display up to 41 evenly spaced rays.
    """
    try:
        from .matplotlib import plot as render
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "matplotlib":
            raise ImportError(
                "Plotting requires Matplotlib. Install it with "
                "python -m pip install 'diffractix[plot]'."
            ) from error
        raise
    return render(result, ax=ax, show_elements=show_elements, show_axis=show_axis)
