"""Schematic 2D plots: install with python -m pip install 'diffractix[plot]'."""
from diffractix import GaussianBeam, Ray, RayBundle, Space, System, ThinLens


def make_plots():
    sources = (
        Ray(x=-0.5e-3, theta=0.015),
        RayBundle.parallel(radius=1.3e-3, count=15),
        GaussianBeam.from_waist(w0=0.3e-3, wavelength=1064e-9),
    )
    figures = []
    for source in sources:
        system = System(
            Space(d=0.06), ThinLens(f=0.08, label="Focus"), Space(d=0.18),
            input=source,
        )
        simulation = system.build()
        result = simulation.run()
        figures.append(result.plot())
    return figures


if __name__ == "__main__":
    make_plots()
    import matplotlib.pyplot as plt
    plt.show()
