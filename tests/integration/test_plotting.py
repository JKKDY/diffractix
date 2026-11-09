"""Plot actual compiled optical results with the public Matplotlib interface."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from diffractix import GRIN, GaussianBeam, Ray, RayBundle, Space, System, ThinLens
from diffractix.plotting import plot
from diffractix.plotting.sampling import sample


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("kind", ["ray", "bundle", "beam"])
def test_result_plot_uses_existing_axes_and_preserves_simulation(kind, monkeypatch):
    sources = {"ray": Ray(x=0.001, theta=0.01),
               "bundle": RayBundle(x=np.array([0.001, -0.001]), theta=np.array([0.01, -0.01])),
               "beam": GaussianBeam.from_waist(0.0003, 1.064e-6)}
    simulation = System(Space(d=0.02), ThinLens(f=0.1), Space(d=0.03),
                        optical_input=sources[kind]).build()
    result = simulation.run()
    before = np.array(result.z, copy=True)
    fig, ax = plt.subplots()
    ax.set_title("User title")
    ax.set_xlim(-0.1, 0.2)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("plot must not show figures"))
    assert result.plot(ax=ax) == (fig, ax)
    assert ax.lines
    assert ax.get_title() == "User title"
    np.testing.assert_allclose(ax.get_xlim(), [-0.1, 0.2])
    np.testing.assert_array_equal(result.z, before)
    np.testing.assert_array_equal(simulation.run().z, before)


def test_ray_samples_follow_physical_angles_on_both_sides_of_lens():
    lens = ThinLens(f=0.1)
    result = System(Space(d=0.02), lens, Space(d=0.03),
                    optical_input=Ray(x=0.001, theta=0.01)).build().run()
    kind, _, _, segments = sample(result)
    assert kind == "ray"
    entrance, exit = segments
    np.testing.assert_allclose(entrance[1], 0.001 + entrance[0]*0.01)
    height = 0.0012
    angle = 0.01 - height/0.1
    np.testing.assert_allclose(exit[1], height+(exit[0]-0.02)*angle)
    assert exit[1][-1] == pytest.approx(result.final.x)


def test_empty_path_plot_does_not_invent_trajectory():
    result = System(optical_input=Ray(x=0.001, theta=0.01)).build().run()
    assert sample(result)[3] == []
    _, ax = plot(result)
    assert not ax.lines or all(len(line.get_xdata()) <= 2 for line in ax.lines)
    assert ax.collections



def test_thick_grin_plot_keeps_boundaries_without_inventing_interior_samples():
    grin = GRIN(d=0.03, g=2, n=1)
    result = System(Space(d=0.02), grin, Space(d=0.04),
                    optical_input=Ray(x=0.001, theta=0.01, n=1)).build().run()
    kind, boundaries_z, boundaries_x, segments = sample(result)
    assert kind == "ray"
    assert len(segments) == 2
    np.testing.assert_allclose(boundaries_z, [0, 0.02, 0.05, 0.09])
    np.testing.assert_allclose(boundaries_x, [state.x for state in result.states])
    for (z, x), state_index, limits in zip(segments, (0, 2), ((0, 0.02), (0.05, 0.09))):
        np.testing.assert_allclose(z[[0, -1]], limits, atol=1e-14)
        assert not np.any((z > 0.02+1e-14) & (z < 0.05-1e-14))
        state = result.states[state_index]
        np.testing.assert_allclose(x, state.x + state.theta*(z-z[0]), atol=1e-14)
    _, ax = plot(result)
    assert "Boundary states only" in ax.get_legend_handles_labels()[1]
    # Marker coordinates include both actual GRIN boundary states.
    markers = np.asarray(ax.collections[0].get_offsets())
    np.testing.assert_allclose(markers[:, 0], boundaries_z)
    np.testing.assert_allclose(markers[:, 1], boundaries_x*1e3)


def test_gaussian_samples_follow_analytic_envelope_and_plot_in_millimetres():
    waist, wavelength = 3e-4, 1.064e-6
    result = System(Space(d=0.05), optical_input=GaussianBeam.from_waist(waist, wavelength)).build().run()
    kind, _, _, segments = sample(result)
    assert kind == "beam"
    (z, width), = segments
    rayleigh = np.pi * waist**2 / wavelength
    np.testing.assert_allclose(z[[0, -1]], [0, 0.05], atol=1e-15)
    np.testing.assert_allclose(width, waist * np.sqrt(1 + (z / rayleigh)**2), rtol=1e-12)
    _, ax = plot(result, show_elements=False, show_axis=False)
    plotted = np.concatenate([np.abs(line.get_ydata()) for line in ax.lines])
    assert plotted.max() == pytest.approx(width.max() * 1e3, rel=1e-12)


def test_plot_without_axes_creates_and_returns_a_new_figure():
    result = System(Space(d=0.02), optical_input=Ray(x=0.001, theta=0.01)).build().run()
    existing = plt.figure()
    fig, ax = plot(result)
    assert fig is not existing and ax.figure is fig
    assert result.plot()[0] is not fig


@pytest.mark.parametrize("show_elements", [False, True])
@pytest.mark.parametrize("show_axis", [False, True])
def test_element_and_axis_overlays_are_optional(show_elements, show_axis):
    lens = ThinLens(f=0.1, label="Objective")
    result = System(Space(d=0.05), lens, Space(d=0.05),
                    optical_input=Ray(x=0.001, theta=0.0)).build().run()
    _, ax = plot(result, show_elements=show_elements, show_axis=show_axis)
    labels = [text.get_text() for text in ax.texts]
    assert ("Objective" in labels) is show_elements
    axis_lines = [line for line in ax.lines
                  if len(line.get_ydata()) and np.all(np.asarray(line.get_ydata(), dtype=float) == 0)]
    assert bool(axis_lines) is show_axis


@pytest.mark.parametrize("count", [5, 41, 42, 100])
def test_bundle_plot_shows_at_most_41_rays(count):
    result = System(Space(d=0.05), optical_input=RayBundle.parallel(radius=1e-3, count=count)).build().run()
    _, ax = plot(result, show_elements=False, show_axis=False)
    assert len(ax.lines) == min(count, 41)
