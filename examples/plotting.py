"""Plot the same lens system for a ray, a ray bundle and a Gaussian beam.

Matplotlib is installed with Diffractix. Plotting returns (figure, axes) and
never calls show(), so figures can be customised or saved before display.
"""

from diffractix import GaussianBeam, Ray, RayBundle, Space, System, ThinLens
from diffractix.units import mm, mrad, nm


def make_plots():
    sources = {
        "Single ray": Ray(x=-0.5 * mm, theta=15 * mrad),
        "Parallel ray bundle": RayBundle.parallel(radius=1.3 * mm, count=15),
        "Gaussian beam": GaussianBeam.from_waist(w0=0.3 * mm, wavelength=1064 * nm),
    }
    figures = []
    for title, source in sources.items():
        system = System(
            Space(d=60 * mm),
            ThinLens(f=80 * mm, label="Focus"),
            Space(d=180 * mm),
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
