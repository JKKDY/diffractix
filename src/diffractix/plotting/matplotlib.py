"""Small Matplotlib renderer for concrete paraxial simulation results."""
import numpy as np
import matplotlib.pyplot as plt

from .sampling import sample


# Legend name, colour, line style and width of each thin element kind.
_STYLES = {
    "ThinLens": ("Thin lens", "#587ca0", "-", 1.3),
    "GaussianAperture": ("Aperture", "#b07852", "--", 1.2),
    "Interface": ("Surface", "#879798", "-", 0.9),
    "Mirror": ("Mirror", "#777f92", "--", 1.2),
}
_DEFAULT_STYLE = ("#a3abb4", ":", 0.8)


def _elements(ax, z, element_data):
    """Draw element markers and name each top-level element once on a top axis.

    Elements of a composite are drawn individually inside one shaded block and
    named once by the composite's label. Every drawn kind gets one legend entry.
    """
    names = {}    # position -> names of top-level elements there
    groups = {}   # composite label -> [start, end]
    shown = set()
    has_unknown_interior = False

    def legend_label(name):
        if name in shown:
            return "_nolegend_"
        shown.add(name)
        return name

    for index, info in enumerate(element_data):
        start, end = float(z[index]), float(z[index + 1])
        composite = info.path[0] if info.path else None
        if composite is not None:
            span = groups.setdefault(composite, [start, end])
            span[0], span[1] = min(span[0], start), max(span[1], end)
        if info.type_name == "Space":
            if start != end and composite is None:
                ax.axvspan(start, end, color="#8b99aa", alpha=0.045, linewidth=0, zorder=0)
            continue
        if start != end:
            ax.axvspan(start, end, facecolor="#b9a48e", alpha=0.18, linewidth=0,
                       label=legend_label("Boundary states only"), zorder=0)
            for position in (start, end):
                ax.axvline(position, color="#9a8b7d", linewidth=0.7, linestyle=":", zorder=2)
            has_unknown_interior = True
            position = (start + end) / 2
        else:
            kind, *style = _STYLES.get(info.type_name, (info.type_name, *_DEFAULT_STYLE))
            color, line_style, width = style
            ax.axvline(start, color=color, linestyle=line_style, linewidth=width, zorder=2,
                       label=legend_label(kind))
            position = start
        if composite is None:
            names.setdefault(position, []).append(info.label or info.type_name)

    for label, (start, end) in groups.items():
        if start != end:
            ax.axvspan(start, end, facecolor="#7d8fa3", alpha=0.12, linewidth=0, zorder=0,
                       label=legend_label("Composite element"))
        names.setdefault((start + end) / 2, []).append(label)

    if names:
        # Names closer than 1 % of the layout would overlap: write them as one label.
        span = max(float(z[-1] - z[0]), 1e-12)
        clusters = []
        for position in sorted(names):
            if clusters and position - clusters[-1][-1] < 0.01 * span:
                clusters[-1].append(position)
            else:
                clusters.append([position])
        positions = [sum(cluster) / len(cluster) for cluster in clusters]
        labels = [" / ".join(dict.fromkeys(name for p in cluster for name in names[p])) for cluster in clusters]
        top = ax.secondary_xaxis("top")
        top.set_xticks(positions, labels)
        top.tick_params(axis="x", length=0, labelsize=8, labelcolor="#5f6b77", rotation=45)
        top.spines["top"].set_visible(False)
        for text in top.get_xticklabels():
            text.set_horizontalalignment("left")
            text.set_rotation_mode("anchor")
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

    unknown = _elements(ax, z, result.element_data) if show_elements else False
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
    if (show_elements or unknown) and ax.get_legend_handles_labels()[1]:
        ax.legend(frameon=False, fontsize=8, loc="best")
    return fig, ax
