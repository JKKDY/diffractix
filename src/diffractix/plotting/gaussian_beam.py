"""Matplotlib diagnostic for a single Gaussian beam."""
import numpy as np

from . import _pyplot

plt = _pyplot()


def _plot_gaussian_beam(source, span, points):
    beam = source._for_display()
    if beam is None:
        raise ValueError(
            f"{source._describe_fields()} cannot be plotted: a field has no value yet. "
            "Give it one (for example n=1.0), or plot a result with result.plot()."
        )

    waist_position = float(beam.z0)
    if span is None:
        scale = max(abs(waist_position), float(beam.z_r))
        span = 4 * scale if scale > 0 else 0.1
    z = np.linspace(-span / 2, span / 2, points)
    radius = np.array([beam.w_at_z(position) for position in z])
    label_height = 0.1 * radius.max()

    figure, axes = plt.subplots(figsize=(8, 4))
    axes.fill_between(z, radius, -radius, color="blue", alpha=0.1, label="Beam Envelope (1/e²)")
    axes.plot(z, radius, "b-", lw=1.5)
    axes.plot(z, -radius, "b-", lw=1.5)
    axes.axvline(0, color="red", linestyle="--", label="Current Position")
    axes.text(0, label_height, " You Are Here", color="red", rotation=90, va="bottom")

    if z.min() < waist_position < z.max():
        axes.axvline(waist_position, color="green", linestyle=":", label="Waist Location")
        axes.text(
            waist_position, label_height, f" Waist (w0={beam.w0 * 1e6:.1f}um)",
            color="green", rotation=90, va="bottom",
        )

    axes.set_title(f"Beam Diagnostic (n={beam.n})")
    axes.set_xlabel("Relative Propagation Distance [m]\n(z > 0 is forward)")
    axes.set_ylabel("Beam Radius w(z) [m]")
    axes.legend(loc="upper right")
    axes.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    plt.show()
