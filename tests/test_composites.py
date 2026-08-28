import pytest

from diffractix.composites import FourF, Slab, ThickLens, Telescope
from diffractix.core.system_vars import AMBIENT_N
from diffractix.graph import InputNode, Parameter


# ----
# SLAB
# ----

def test_slab_structure():
    slab = Slab(d=0.01, n=1.5)

    assert slab.element_names == ("front", "body", "back")
    assert slab.elements == (
        slab.front,
        slab.body,
        slab.back,
    )


def test_slab_parameter_wiring():
    slab = Slab(d=0.01, n=1.5)

    assert slab.front.n2.node is slab.n
    assert slab.body.d.node is slab.d
    assert slab.body.n.node is slab.n
    assert slab.back.n1.node is slab.n


def test_slab_uses_system_ambient_index_by_default():
    slab = Slab(d=0.01, n=1.5)

    assert slab.n_ambient.node is AMBIENT_N
    assert slab.front.n1.node is slab.n_ambient
    assert slab.back.n2.node is slab.n_ambient


def test_slab_accepts_explicit_ambient_index():
    slab = Slab(d=0.01, n=1.5, n_ambient=1.33)

    assert isinstance(slab.n_ambient.node, Parameter)
    assert slab.n_ambient.value == pytest.approx(1.33)
    assert slab.front.n1.node is slab.n_ambient
    assert slab.back.n2.node is slab.n_ambient


def test_slab_parameter_reassignment_propagates():
    slab = Slab(d=0.01, n=1.5)

    slab.d = 0.02
    slab.n = 1.6

    assert slab.body.d.value == pytest.approx(0.02)
    assert slab.front.n2.value == pytest.approx(1.6)
    assert slab.body.n.value == pytest.approx(1.6)
    assert slab.back.n1.value == pytest.approx(1.6)


def test_slab_ambient_reassignment_propagates():
    slab = Slab(d=0.01, n=1.5)

    slab.n_ambient = 1.33

    assert slab.front.n1.value == pytest.approx(1.33)
    assert slab.back.n2.value == pytest.approx(1.33)


# ----------
# THICK LENS
# ----------

def test_thick_lens_structure():
    lens = ThickLens(d=0.01, n=1.5, R1=0.1, R2=-0.1)

    assert lens.element_names == ("front", "body", "back")
    assert lens.elements == (
        lens.front,
        lens.body,
        lens.back,
    )


def test_thick_lens_parameter_wiring():
    lens = ThickLens(d=0.01, n=1.5, R1=0.1, R2=-0.1)

    assert lens.front.n2.node is lens.n
    assert lens.front.R.node is lens.R1
    assert lens.body.d.node is lens.d
    assert lens.body.n.node is lens.n
    assert lens.back.n1.node is lens.n
    assert lens.back.R.node is lens.R2


def test_thick_lens_uses_system_ambient_index_by_default():
    lens = ThickLens(d=0.01, n=1.5)

    assert lens.n_ambient.node is AMBIENT_N
    assert lens.front.n1.node is lens.n_ambient
    assert lens.back.n2.node is lens.n_ambient


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


def test_thick_lens_ambient_reassignment_propagates():
    lens = ThickLens(d=0.01, n=1.5)

    lens.n_ambient = 1.33

    assert lens.front.n1.value == pytest.approx(1.33)
    assert lens.back.n2.value == pytest.approx(1.33)


# ------
# FOUR F
# ------

def test_four_f_structure():
    four_f = FourF(f1=0.1, f2=0.2)

    assert four_f.element_names == (
        "input_space",
        "lens1",
        "fourier_space",
        "lens2",
        "output_space",
    )

    assert four_f.elements == (
        four_f.input_space,
        four_f.lens1,
        four_f.fourier_space,
        four_f.lens2,
        four_f.output_space,
    )


def test_four_f_parameter_wiring():
    four_f = FourF(f1=0.1, f2=0.2)

    assert four_f.input_space.d.node is four_f.f1
    assert four_f.lens1.f.node is four_f.f1
    assert four_f.lens2.f.node is four_f.f2
    assert four_f.output_space.d.node is four_f.f2


def test_four_f_middle_space_is_sum_of_focal_lengths():
    four_f = FourF(f1=0.1, f2=0.2)

    assert four_f.fourier_space.d.value == pytest.approx(0.3)


def test_four_f_reassignment_updates_all_linked_elements():
    four_f = FourF(f1=0.1, f2=0.2)

    four_f.f1 = 0.15
    four_f.f2 = 0.25

    assert four_f.input_space.d.value == pytest.approx(0.15)
    assert four_f.lens1.f.value == pytest.approx(0.15)
    assert four_f.fourier_space.d.value == pytest.approx(0.4)
    assert four_f.lens2.f.value == pytest.approx(0.25)
    assert four_f.output_space.d.value == pytest.approx(0.25)


# ---------
# TELESCOPE
# ---------

def test_telescope_structure():
    telescope = Telescope(f1=0.1, f2=0.2)

    assert telescope.element_names == (
        "lens1",
        "space",
        "lens2",
    )

    assert telescope.elements == (
        telescope.lens1,
        telescope.space,
        telescope.lens2,
    )


def test_telescope_parameter_wiring():
    telescope = Telescope(f1=0.1, f2=0.2)

    assert telescope.lens1.f.node is telescope.f1
    assert telescope.lens2.f.node is telescope.f2


def test_telescope_spacing_is_sum_of_focal_lengths():
    telescope = Telescope(f1=0.1, f2=0.2)

    assert telescope.space.d.value == pytest.approx(0.3)


def test_telescope_reassignment_updates_spacing_and_lenses():
    telescope = Telescope(f1=0.15, f2=0.25)

    telescope.f1 = 0.2
    telescope.f2 = 0.3

    assert telescope.lens1.f.value == pytest.approx(0.2)
    assert telescope.space.d.value == pytest.approx(0.5)
    assert telescope.lens2.f.value == pytest.approx(0.3)


# --------------------
# VARIABLE PARAMETERS
# --------------------

def test_slab_composite_parameter_can_be_variable():
    slab = Slab(d=0.01, n=1.5)

    slab.variable("n")

    assert slab.n.node.is_variable
    assert slab.front.n2.node is slab.n
    assert slab.body.n.node is slab.n
    assert slab.back.n1.node is slab.n


def test_thick_lens_composite_parameter_can_be_variable():
    lens = ThickLens(d=0.01, n=1.5)

    lens.variable("R1")

    assert lens.R1.node.is_variable
    assert lens.front.R.node is lens.R1


def test_four_f_composite_parameter_can_be_variable():
    four_f = FourF(f1=0.1, f2=0.2)

    four_f.variable("f1")

    assert four_f.f1.node.is_variable
    assert four_f.input_space.d.node is four_f.f1
    assert four_f.lens1.f.node is four_f.f1