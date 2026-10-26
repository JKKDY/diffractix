"""Small Matplotlib renderer for concrete paraxial simulation results."""
import numpy as np
import matplotlib.pyplot as plt

from .sampling import sample


def _elements(ax, z, element_info):
    labels = {}
    has_unknown_interior = False
    for index, info in enumerate(element_info):
        start, end = z[index:index + 2]
        if info.type_name == "Space":
            if start != end:
                ax.axvspan(start, end, color="#8b99aa", alpha=0.045, linewidth=0, zorder=0)
            continue
        if start != end:
            ax.axvspan(
                start, end, facecolor="#b9a48e", alpha=0.18, linewidth=0,
                label="Boundary states only" if not has_unknown_interior else "_nolegend_",
                zorder=0,
            )
            for position in (start, end):
                ax.axvline(position, color="#9a8b7d", linewidth=0.7, linestyle=":", zorder=2)
            has_unknown_interior = True
            position = (start + end) / 2
        else:
            color, style, width = {
                "ThinLens": ("#587ca0", "-", 1.3),
                "GaussianAperture": ("#b07852", "--", 1.2),
                "Interface": ("#879798", "-", 0.9),
                "Mirror": ("#777f92", "--", 1.2),
            }.get(info.type_name, ("#a3abb4", ":", 0.8))
            ax.axvline(start, color=color, linestyle=style, linewidth=width, zorder=2)
            position = start
        name = info.path or info.label or info.type_name
        labels.setdefault(float(position), []).append(name)

    # Keep dense optical layouts readable; all transformations still have markers.
    if len(labels) <= 12:
        for position, names in labels.items():
            ax.text(
                position, 0.97, " / ".join(dict.fromkeys(names)),
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="top", fontsize=8, color="#7f8994", alpha=0.85,
            )
    return has_unknown_interior


def plot(result, ax=None, *, show_elements=True, show_axis=True):
    kind, z, boundaries, segments = sample(result)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    else:
        fig = ax.figure
    color = "#df7664" if kind == "beam" else "#377cb1"
    label = "Beam envelope" if kind == "beam" else "Ray trajectories"
    alpha = 0.5 if kind == "bundle" and boundaries.shape[1] > 10 else 0.85
    for index, (positions, values) in enumerate(segments):
        values = values * 1e3
        if kind == "beam":
            ax.fill_between(positions, -values, values, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(positions, values, color=color, linewidth=1.5, zorder=3, label=label if index == 0 else "_nolegend_")
            ax.plot(positions, -values, color=color, linewidth=1.5, zorder=3)
        else:
            lines = ax.plot(positions, values, color=color, alpha=alpha, linewidth=1.1, zorder=3)
            if index == 0 and lines:
                lines[0].set_label(label)

    # Separate markers expose boundary jumps and unmodelled interiors honestly.
    if kind == "beam":
        positions = np.concatenate((z, z))
        transverse = np.concatenate((boundaries, -boundaries))
    elif kind == "bundle":
        positions = np.repeat(z, boundaries.shape[1])
        transverse = boundaries.ravel()
    else:
        positions, transverse = z, boundaries
    ax.scatter(positions, transverse * 1e3, color=color, s=8, alpha=alpha, linewidths=0, zorder=4)

    unknown = _elements(ax, z, result.element_info) if show_elements else False
    if show_axis:
        ax.axhline(0, color="#8a929b", linewidth=0.8, linestyle="--", alpha=0.65, zorder=1)
    ax.set_xlabel("Optical axis z [m]")
    ax.set_ylabel("Beam radius ±w [mm]" if kind == "beam" else "Transverse position x [mm]")
    titles = {"beam": "Gaussian beam envelope", "ray": "Ray propagation", "bundle": "Ray bundle propagation"}
    title = titles[kind]
    if kind == "bundle" and result.source.size > boundaries.shape[1]:
        title += f" ({boundaries.shape[1]} of {result.source.size} rays)"
    if not any(ax.get_title(loc=loc) for loc in ("left", "center", "right")):
        ax.set_title(title, loc="left", fontsize=11, fontweight="normal")
    ax.grid(True, color="#a0a8b1", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.03, y=0.18)
    if created and np.ptp(z) == 0:
        ax.set_xlim(z[0] - 1e-3, z[0] + 1e-3)
    if unknown:
        ax.legend(frameon=False, fontsize=8)
    return fig, ax
