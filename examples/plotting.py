"""Plot the same lens system for a ray, a ray bundle and a Gaussian beam.

Matplotlib is installed with Diffractix. Plotting returns (figure, axes) and
never calls show(), so figures can be customised or saved before display.
"""

from diffractix import GaussianBeam, Ray, RayBundle, Space, System, ThinLens


def make_plots():
    sources = {
        "Single ray": Ray(x=-0.5e-3, theta=0.015),
        "Parallel ray bundle": RayBundle.parallel(radius=1.3e-3, count=15),
        "Gaussian beam": GaussianBeam.from_waist(w0=0.3e-3, wavelength=1064e-9),
    }
    figures = []
    for title, source in sources.items():
        system = System(
            Space(d=0.06),
            ThinLens(f=0.08, label="Focus"),
            Space(d=0.18),
            optical_input=source,
        )
        figure, axes = system.build().run().plot()
        axes.set_title(title)
        figures.append(figure)
    return figures


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    make_plots()
    plt.show()
