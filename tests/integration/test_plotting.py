"""Physical sampling and the Matplotlib result interface."""
from dataclasses import dataclass

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from diffractix import (
    ABCD, GaussianAperture, GaussianBeam, GRIN, Interface, Parameter,
    ParaxialState, Ray, RayBundle, Slab, Space, System, ThinLens,
)
from diffractix.backend import grad
from diffractix.plotting import plot
from diffractix.plotting.sampling import sample


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("kind", ["ray", "bundle", "beam"])
def test_plot_entry_points_and_existing_axes(kind, monkeypatch):
    source = {
        "ray": Ray(x=1e-3, theta=0.01),
        "bundle": RayBundle.parallel(radius=1e-3, count=5),
        "beam": GaussianBeam.from_waist(0.3e-3, 1064e-9),
    }[kind]
    result = System(Space(d=0.1), input=source).build().run()
    def forbidden(*args, **kwargs):
        raise AssertionError("Plotting must not show or save figures")
    monkeypatch.setattr(plt, "show", forbidden)
    before = dict(matplotlib.rcParams)
    first, axes = result.plot(show_elements=False, show_axis=False)
    assert axes.figure is first
    assert len(axes.lines) == (2 if kind == "beam" else 5 if kind == "bundle" else 1)
    assert axes.get_xlabel() == "Optical axis z [m]"
    assert axes.get_ylabel().endswith("[mm]")
    if kind == "beam":
        upper, lower = axes.lines
        np.testing.assert_allclose(upper.get_ydata(), -np.asarray(lower.get_ydata()))
        assert len(axes.collections) == 2  # Envelope fill and boundary markers.
    assert dict(matplotlib.rcParams) == before

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-0.1, 0.3)
    ax.set_title("My optical layout", loc="left")
    assert plot(result, ax=ax) == (fig, ax)
    np.testing.assert_allclose(fig.get_size_inches(), [6, 3])
    np.testing.assert_allclose(ax.get_xlim(), [-0.1, 0.3])
    assert ax.get_title(loc="left") == "My optical layout"
    assert result.plot(ax=ax, show_elements=False) == (fig, ax)


def test_ray_sampling_uses_the_entrance_state_on_each_side_of_a_lens():
    lens = ThinLens(f=0.03)
    result = System(Space(d=0.02), lens, Space(d=0.04), input=Ray(x=1e-3, theta=0.01)).build().run()
    kind, _, _, segments = sample(result)
    assert kind == "ray"
    first, second = segments
    np.testing.assert_allclose(first[1], 1e-3 + 0.01 * first[0])
    height = 1e-3 + 0.02 * 0.01
    angle = 0.01 - height / 0.03
    np.testing.assert_allclose(second[1], height + angle * (second[0] - 0.02))
    np.testing.assert_allclose(second[1][-1], result.final.x, atol=1e-15)


@pytest.mark.parametrize("index", [1.0, 1.5])
def test_gaussian_sampling_captures_the_waist_in_the_local_medium(index):
    waist, location, wavelength = 30e-6, 0.037, 1064e-9
    source = GaussianBeam.from_waist(waist, wavelength, z_waist_loc=location, n=index)
    result = System(Space(d=0.1), input=source, ambient_n=index).build().run()
    _, _, _, [(positions, widths)] = sample(result)
    rayleigh = np.pi * index * waist**2 / wavelength
    expected = waist * np.sqrt(1 + ((positions - location) / rayleigh)**2)
    np.testing.assert_allclose(widths, expected, rtol=1e-12)
    assert positions[np.argmin(widths)] == pytest.approx(location)
    assert widths.min() == pytest.approx(waist)


def test_slab_sampling_uses_refracted_angles_inside_the_medium():
    result = System(
        Space(d=0.01), Slab(d=0.02, n=1.5), Space(d=0.01),
        input=Ray(x=1e-3, theta=0.015),
    ).build().run()
    _, _, _, segments = sample(result)
    assert len(segments) == 3
    inside_z, inside_x = segments[1]
    np.testing.assert_allclose(inside_x, 1e-3 + 0.015 * 0.01 + (inside_z - 0.01) * 0.015 / 1.5)


def test_unknown_thick_elements_have_boundaries_not_interpolated_trajectories():
    result = System(
        Space(d=0.01), GRIN(d=0.02, g=10.0, n=1.0),
        ABCD(A=2.0, D=0.5, thickness=0.03), Space(d=0.04),
        input=Ray(x=1e-3, theta=0.01),
    ).build().run()
    _, z, boundaries, segments = sample(result)
    assert len(segments) == 2
    np.testing.assert_allclose([segments[0][0][-1], segments[1][0][0]], [0.01, 0.06])
    np.testing.assert_allclose(boundaries, result.x)
    np.testing.assert_allclose(z, [0, 0.01, 0.03, 0.06, 0.1])
    _, ax = result.plot(show_axis=False)
    assert any(patch.get_label() == "Boundary states only" for patch in ax.patches)
    for line in ax.lines:
        positions = np.asarray(line.get_xdata())
        if len(positions) > 2:
            assert positions.max() <= 0.01 or positions.min() >= 0.06 - 1e-14


def test_sampling_does_not_guess_space_behavior_from_a_black_box_matrix():
    element = ABCD(B=0.1, thickness=0.1)
    result = System(element, input=Ray(x=0.0, theta=0.01)).build().run()
    assert result.element_info[0].type_name == "ABCD"
    assert sample(result)[3] == []


def test_zero_length_elements_repeated_positions_and_markers():
    lens, aperture, surface = ThinLens(f=0.08), GaussianAperture(a=1e-3), Interface(n1=1.0, n2=1.5)
    result = System(
        Space(d=0.05), lens, aperture, surface, Space(d=0.0), Space(d=0.02),
        input=GaussianBeam.from_waist(0.3e-3, 1064e-9),
    ).build().run()
    _, _, boundaries, segments = sample(result)
    assert len(segments) == 2
    assert boundaries[3] < boundaries[2]  # Aperture jump retained at the same z.
    _, ax = result.plot(show_axis=False)
    markers = [line for line in ax.lines if len(line.get_xdata()) == 2]
    assert len(markers) == 3
    for marker in markers:
        np.testing.assert_allclose(marker.get_xdata(), [0.05, 0.05])
    assert any(line.get_linestyle() == "--" for line in markers)


@pytest.mark.parametrize("count", [0, 1, 7, 101])
def test_bundle_sampling_and_readable_ray_selection(count):
    source = RayBundle(x=np.linspace(-1e-3, 1e-3, count), theta=np.full(count, 0.01))
    result = System(Space(d=0.1), input=source).build().run()
    _, _, boundaries, [(z, values)] = sample(result)
    selected = np.linspace(0, count - 1, min(count, 41), dtype=int)
    assert boundaries.shape == (2, len(selected))
    np.testing.assert_allclose(values, np.asarray(result.source.x)[selected] + z[:, None] * 0.01)
    _, ax = result.plot(show_elements=False, show_axis=False)
    assert len(ax.lines) == len(selected)
    if count > 41:
        assert "41 of 101" in ax.get_title(loc="left")


@pytest.mark.parametrize("kind", ["ray", "beam"])
def test_empty_system_has_boundary_values_without_invented_propagation(kind):
    source = Ray(x=1e-3, theta=0.01) if kind == "ray" else GaussianBeam.from_waist(0.3e-3, 1064e-9)
    result = System(input=source).build().run()
    assert sample(result)[3] == []
    _, ax = result.plot()
    assert len(ax.collections) == 1


def test_plotting_preserves_results_and_future_simulations():
    source = RayBundle.parallel(radius=1e-3, count=7)
    system = System(Space(d=0.03), ThinLens(f=0.05), Space(d=0.08), input=source)
    simulation = system.build()
    result = simulation.run()
    before_z, before_vectors = np.array(result.z, copy=True), np.array(result.vectors, copy=True)
    # Editing the original model must not affect either metadata or sampling.
    system.placements[0].element.label = "Edited outside"
    system.placements[0].element.d = 99.0
    result.plot()
    np.testing.assert_array_equal(result.z, before_z)
    np.testing.assert_array_equal(result.vectors, before_vectors)
    np.testing.assert_array_equal(simulation.run().vectors, before_vectors)
    assert result.element_info[0].label != "Edited outside"


def test_builtin_input_subclasses_keep_the_supported_sampling_interface():
    @dataclass
    class TaggedRay(Ray):
        tag: str = "custom"
    result = System(Space(d=0.02), input=TaggedRay(x=1e-3, theta=0.01)).build().run()
    assert sample(result)[0] == "ray"


def test_unsupported_custom_state_reports_its_type():
    @dataclass
    class Custom(ParaxialState):
        value: float = 1.0
        def propagate(self, A, B, C, D, n):
            return type(self)(value=A * self.value)
    result = System(input=Custom()).build().run()
    with pytest.raises(TypeError, match="supports numerical Ray.*NumericalCustom"):
        result.plot()


def test_sampling_requires_concrete_results_outside_differentiation():
    simulation = System(Space(d=Parameter(0.1).variable()), input=Ray(x=1e-3, theta=0.01)).build()
    with pytest.raises(ValueError, match="concrete.*outside differentiation or JIT"):
        grad(lambda theta: sample(simulation.run(theta))[2].sum())(simulation.initial_values)
