"""Builtin composite wiring and independent aggregate optical matrices."""

import numpy as np
import pytest

from diffractix.composites import FourF, Slab, ThickLens, Telescope
from diffractix.system import AMBIENT_N
from diffractix.graph import Node, Parameter


def aggregate_matrix(composite, bindings=None):
    matrix = np.eye(2)
    for leaf in composite.leaf_elements:
        current = np.array([
            [entry.evaluate(bindings or {}) if isinstance(entry, Node) else entry for entry in row]
            for row in leaf.matrix
        ])
        matrix = current @ matrix
    return matrix


@pytest.mark.parametrize("factory, names", [
    (lambda: Slab(d=0.01, n=1.5), ("front", "body", "back")),
    (lambda: ThickLens(d=0.01, n=1.5, R1=0.1, R2=-0.1), ("front", "body", "back")),
    (lambda: Telescope(f1=0.1, f2=0.2), ("lens1", "space", "lens2")),
    (lambda: FourF(f1=0.1, f2=0.2), ("input_space", "telescope", "output_space")),
], ids=["slab", "thick-lens", "telescope", "four-f"])
def test_builtin_composite_children_are_named_attributes_in_propagation_order(factory, names):
    composite = factory()
    assert composite.element_names == names
    assert composite.elements == tuple(getattr(composite, name) for name in names)


# ----
# SLAB
# ----


def test_slab_parameter_wiring():
    slab = Slab(d=0.01, n=1.5)

    assert slab.front.n2.node is slab.n
    assert slab.body.d.node is slab.d
    assert slab.body.n.node is slab.n
    assert slab.back.n1.node is slab.n


@pytest.mark.parametrize("composite_type", [Slab, ThickLens])
def test_refracting_composites_share_one_ambient_handle_on_both_surfaces(composite_type):
    default = composite_type(d=0.01, n=1.5)
    assert default.ambient_n.node is AMBIENT_N
    explicit = composite_type(d=0.01, n=1.5, ambient_n=1.33)
    assert isinstance(explicit.ambient_n.node, Parameter)
    for composite in (default, explicit):
        assert composite.front.n1.node is composite.ambient_n
        assert composite.back.n2.node is composite.ambient_n
    explicit.ambient_n = 1.4
    assert explicit.front.n1.value == explicit.back.n2.value == pytest.approx(1.4)


def test_slab_parameter_reassignment_propagates():
    slab = Slab(d=0.01, n=1.5)

    slab.d = 0.02
    slab.n = 1.6

    assert slab.body.d.value == pytest.approx(0.02)
    assert slab.front.n2.value == pytest.approx(1.6)
    assert slab.body.n.value == pytest.approx(1.6)
    assert slab.back.n1.value == pytest.approx(1.6)


# ----------
# THICK LENS
# ----------


def test_thick_lens_parameter_wiring():
    lens = ThickLens(d=0.01, n=1.5, R1=0.1, R2=-0.1)

    assert lens.front.n2.node is lens.n
    assert lens.front.R.node is lens.R1
    assert lens.body.d.node is lens.d
    assert lens.body.n.node is lens.n
    assert lens.back.n1.node is lens.n
    assert lens.back.R.node is lens.R2


def test_thick_lens_parameter_reassignment_propagates():
    lens = ThickLens(d=0.01, n=1.5, R1=0.1, R2=-0.1)

    lens.d = 0.02
    lens.n = 1.6
    lens.R1 = 0.2
    lens.R2 = -0.3

    assert lens.body.d.value == pytest.approx(0.02)
    assert lens.front.n2.value == pytest.approx(1.6)
    assert lens.body.n.value == pytest.approx(1.6)
    assert lens.back.n1.value == pytest.approx(1.6)
    assert lens.front.R.value == pytest.approx(0.2)
    assert lens.back.R.value == pytest.approx(-0.3)


# ------
# FOUR F
# ------


def test_four_f_parameter_wiring():
    four_f = FourF(f1=0.1, f2=0.2)

    assert four_f.input_space.d.node is four_f.f1
    assert four_f.telescope.f1.node is four_f.f1
    assert four_f.telescope.f2.node is four_f.f2
    assert four_f.output_space.d.node is four_f.f2


def test_four_f_reassignment_updates_all_linked_elements():
    four_f = FourF(f1=0.1, f2=0.2)

    four_f.f1 = 0.15
    four_f.f2 = 0.25

    assert four_f.input_space.d.value == pytest.approx(0.15)
    assert four_f.telescope.lens1.f.value == pytest.approx(0.15)
    assert four_f.telescope.space.d.value == pytest.approx(0.4)
    assert four_f.telescope.lens2.f.value == pytest.approx(0.25)
    assert four_f.output_space.d.value == pytest.approx(0.25)


# ---------
# TELESCOPE
# ---------


def test_telescope_parameter_wiring():
    telescope = Telescope(f1=0.1, f2=0.2)

    assert telescope.lens1.f.node is telescope.f1
    assert telescope.lens2.f.node is telescope.f2


def test_telescope_reassignment_updates_spacing_and_lenses():
    telescope = Telescope(f1=0.15, f2=0.25)

    telescope.f1 = 0.2
    telescope.f2 = 0.3

    assert telescope.lens1.f.value == pytest.approx(0.2)
    assert telescope.space.d.value == pytest.approx(0.5)
    assert telescope.lens2.f.value == pytest.approx(0.3)


# -----------------
# AGGREGATE OPTICS
# -----------------


@pytest.mark.parametrize("ambient", [1.0, 1.33])
def test_slab_aggregate_matrix_uses_physical_angle_and_returns_to_ambient(ambient):
    slab = Slab(d=0.012, n=1.52)

    np.testing.assert_allclose(
        aggregate_matrix(slab, {AMBIENT_N.key: ambient}),
        [[1.0, 0.012 * ambient / 1.52], [0.0, 1.0]],
        atol=1e-15,
    )
    assert slab.element_length.evaluate() == pytest.approx(0.012)
    assert slab.back.refractive_index.evaluate({AMBIENT_N.key: ambient}) == ambient


@pytest.mark.parametrize("radii", [(0.08, -0.12), (-0.08, 0.12), (np.inf, np.inf)])
def test_thick_lens_aggregate_matrix_matches_independent_surface_power(radii):
    thickness, index, ambient = 0.008, 1.52, 1.33
    first_radius, second_radius = radii
    lens = ThickLens(d=thickness, n=index, R1=first_radius, R2=second_radius, ambient_n=ambient)
    entrance = np.array([[1.0, 0.0], [(ambient - index) / (index * first_radius), ambient / index]])
    drift = np.array([[1.0, thickness], [0.0, 1.0]])
    exit = np.array([[1.0, 0.0], [(index - ambient) / (ambient * second_radius), index / ambient]])

    np.testing.assert_allclose(aggregate_matrix(lens), exit @ drift @ entrance, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("focal_lengths", [(0.1, 0.2), (0.2, 0.1)])
def test_telescope_is_afocal_with_expected_angular_magnification(focal_lengths):
    f1, f2 = focal_lengths
    telescope = Telescope(f1=f1, f2=f2)

    np.testing.assert_allclose(
        aggregate_matrix(telescope),
        [[-f2 / f1, f1 + f2], [0.0, -f1 / f2]],
        atol=1e-14,
    )


@pytest.mark.parametrize("focal_lengths", [(0.1, 0.2), (0.2, 0.1)])
def test_four_f_images_plane_with_expected_transverse_magnification(focal_lengths):
    f1, f2 = focal_lengths
    relay = FourF(f1=f1, f2=f2)

    np.testing.assert_allclose(
        aggregate_matrix(relay),
        [[-f2 / f1, 0.0], [0.0, -f1 / f2]],
        atol=1e-14,
    )


def test_shared_focal_parameter_has_one_degree_of_freedom_after_reassignment():
    from diffractix.graph import compile_graph

    focal_length = Parameter(0.1).variable()
    relay = FourF(f1=focal_length, f2=2 * focal_length)
    roots = (relay.input_space.d, relay.telescope.space.d, relay.output_space.d)
    program = compile_graph(roots)

    assert program.parameter_indices == {id(focal_length): 0}
    np.testing.assert_allclose(program.evaluate([0.2]), [0.2, 0.6, 0.4])
    focal_length.fixed()
    focal_length.value = 0.15
    np.testing.assert_allclose([root.evaluate() for root in roots], [0.15, 0.45, 0.3])
    np.testing.assert_allclose(program.evaluate([0.2]), [0.2, 0.6, 0.4])


@pytest.mark.parametrize("composite_type, kwargs", [
    (Slab, dict(d=0.01, n=1.5)),
    (ThickLens, dict(d=0.01, n=1.5)),
    (Telescope, dict(f1=0.1, f2=0.2)),
    (FourF, dict(f1=0.1, f2=0.2)),
])
def test_default_composite_labels_are_unique_and_explicit_labels_are_kept(composite_type, kwargs):
    first, second = composite_type(**kwargs), composite_type(**kwargs)
    assert first.label != second.label
    assert first.label.startswith(composite_type.__name__)
    named = composite_type(**kwargs, label="mine")
    assert named.label == "mine"
    assert all(part.label.startswith("mine") for part in named.leaf_elements)
