import numpy as np

from diffractix.elements import (
    ThinLens,
    Space,
    Mirror,
    Interface,
    ABCD,
    GaussianAperture,
)
from diffractix.graph import (
    Node,
    Literal,
    Parameter,
    InputNode,
    SystemVar,
    compile_ast,
)


def as_node(value):
    """Normalize a declarative scalar into an AST node."""
    return value if isinstance(value, Node) else Literal(value)


def evaluate_matrix(element, context=None):
    """Compile and evaluate an element's declarative ABCD matrix."""
    roots = [
        as_node(value)
        for row in element.matrix
        for value in row
    ]

    compiled = compile_ast(roots, context)
    values = compiled.evaluate(compiled.initial_values)

    return values.reshape(2, 2)


# ---------
# THIN LENS
# ---------
def test_thin_lens_declaration():
    lens = ThinLens(f=0.1, label="Objective")

    assert lens.parameter_names == ("f",)
    assert lens.label == "Objective"
    assert isinstance(lens.f, InputNode)
    assert isinstance(lens.f.node, Parameter)
    assert lens.f.value == 0.1

    assert lens.element_length == 0.0
    assert lens.element_refractive_index is None


def test_thin_lens_matrix():
    for f in [-0.5, 0.1, 1.0, np.inf]:
        lens = ThinLens(f=f)

        expected = np.array([
            [1.0, 0.0],
            [-1.0 / f, 1.0],
        ])

        np.testing.assert_allclose(
            evaluate_matrix(lens),
            expected,
        )


def test_thin_lens_matrix_tracks_input_handle():
    """
    Matrix expressions should retain the stable input handle until compilation.
    """
    lens = ThinLens(f=0.1)
    matrix = lens.matrix

    lens.f = 0.2

    assert matrix[1][0].value == -5.0


# -----
# SPACE
# -----
def test_space_declaration():
    space = Space(d=10.0, n=1.5, label="GlassBlock")

    assert space.parameter_names == ("d", "n")
    assert isinstance(space.d, InputNode)
    assert isinstance(space.n, InputNode)

    assert space.d.value == 10.0
    assert space.n.value == 1.5

    assert space.element_length is space.d
    assert space.element_refractive_index is space.n


def test_space_matrix():
    for d in [0.1, 1.0, 10.0]:
        space = Space(d=d, n=1.5)

        expected = np.array([
            [1.0, d],
            [0.0, 1.0],
        ])

        np.testing.assert_allclose(
            evaluate_matrix(space),
            expected,
        )


def test_space_can_request_refractive_index_inheritance():
    space = Space(d=1.0)

    assert isinstance(space.n, InputNode)
    assert space.n.node is None
    assert space.element_refractive_index is space.n


def test_space_refractive_index_does_not_affect_matrix():
    air = Space(d=1.0, n=1.0)
    glass = Space(d=1.0, n=1.5)

    np.testing.assert_allclose(
        evaluate_matrix(air),
        evaluate_matrix(glass),
    )


# ------
# MIRROR
# ------
def test_mirror_declaration():
    mirror = Mirror(R=0.5, label="M1")

    assert mirror.parameter_names == ("R",)
    assert mirror.R.value == 0.5
    assert mirror.element_length == 0.0
    assert mirror.element_refractive_index is None


def test_mirror_matrix():
    for radius in [0.5, -0.5, np.inf]:
        mirror = Mirror(R=radius)

        expected = np.array([
            [1.0, 0.0],
            [-2.0 / radius, 1.0],
        ])

        np.testing.assert_allclose(
            evaluate_matrix(mirror),
            expected,
        )


# ---------
# INTERFACE
# ---------
def test_interface_declaration():
    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=0.1,
        label="FrontSurface",
    )

    assert interface.parameter_names == ("n1", "n2", "R")
    assert interface.element_length == 0.0
    assert interface.element_refractive_index is interface.n2


def test_interface_matrix():
    n1 = 1.0
    n2 = 1.5
    radius = 0.05

    interface = Interface(
        n1=n1,
        n2=n2,
        R=radius,
    )

    power = (n1 - n2) / (radius * n2)

    expected = np.array([
        [1.0, 0.0],
        [power, n1 / n2],
    ])

    np.testing.assert_allclose(
        evaluate_matrix(interface),
        expected,
    )


def test_flat_interface_has_zero_power():
    interface = Interface(
        n1=1.0,
        n2=1.5,
        R=np.inf,
    )

    matrix = evaluate_matrix(interface)

    assert matrix[1, 0] == 0.0
    assert matrix[1, 1] == 1.0 / 1.5


# ----------
# ABCD
# ----------
def test_abcd_declaration():
    element = ABCD(
        A=2.0,
        D=0.5,
        thickness=0.1,
        n=1.5,
        label="Relay",
    )

    assert element.parameter_names == (
        "A",
        "B",
        "C",
        "D",
        "thickness",
        "n",
    )

    assert element.A.value == 2.0
    assert element.B.value == 0.0
    assert element.C.value == 0.0
    assert element.D.value == 0.5

    assert element.element_length is element.thickness
    assert element.element_refractive_index is element.n


def test_abcd_matrix_is_declarative():
    element = ABCD(
        A=2.0,
        B=0.5,
        C=0.1,
        D=1.0,
    )

    matrix = element.matrix

    assert matrix[0][0] is element.A
    assert matrix[0][1] is element.B
    assert matrix[1][0] is element.C
    assert matrix[1][1] is element.D

    np.testing.assert_allclose(
        evaluate_matrix(element),
        [[2.0, 0.5], [0.1, 1.0]],
    )


def test_abcd_matrix_override():
    matrix = np.array([
        [2.0, 0.5],
        [0.1, 1.0],
    ])

    element = ABCD(
        matrix_val=matrix,
        thickness=0.1,
    )

    np.testing.assert_allclose(
        evaluate_matrix(element),
        matrix,
    )

    assert element.thickness.value == 0.1


def test_abcd_matrix_setter_preserves_handles():
    element = ABCD()

    handles = (
        element.A,
        element.B,
        element.C,
        element.D,
    )

    element.matrix = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    assert element.A is handles[0]
    assert element.B is handles[1]
    assert element.C is handles[2]
    assert element.D is handles[3]

    np.testing.assert_allclose(
        evaluate_matrix(element),
        [[2.0, 3.0], [4.0, 5.0]],
    )


def test_abcd_matrix_override_rejects_wrong_shape():
    element = ABCD()

    with np.testing.assert_raises(ValueError):
        element.matrix = np.eye(3)


def test_abcd_can_request_refractive_index_inheritance():
    element = ABCD(thickness=1.0)

    assert isinstance(element.n, InputNode)
    assert element.n.node is None
    assert element.element_refractive_index is element.n


def test_abcd_explicit_refractive_index():
    element = ABCD(n=1.5)

    assert element.n.value == 1.5
    assert element.element_refractive_index is element.n


def test_abcd_refractive_index_can_be_context_dependent():
    glass_n = SystemVar("glass_n")
    element = ABCD(n=glass_n)

    assert element.n.node is glass_n

    compiled = compile_ast(
        [element.element_refractive_index],
        context={"glass_n": 1.5},
    )

    result = compiled.evaluate(compiled.initial_values)

    np.testing.assert_allclose(result, [1.5])


# -----------------
# GAUSSIAN APERTURE
# -----------------
def test_gaussian_aperture_declaration():
    aperture = GaussianAperture(a=1e-3)

    assert aperture.parameter_names == ("a", "wavelength")
    assert aperture.a.value == 1e-3

    assert isinstance(aperture.wavelength, InputNode)
    assert isinstance(aperture.wavelength.node, SystemVar)
    assert aperture.wavelength.node.name == "wavelength"

    assert aperture.element_length == 0.0
    assert aperture.element_refractive_index is None


def test_gaussian_aperture_matrix():
    radius = 1e-3
    wavelength = 1064e-9

    aperture = GaussianAperture(a=radius)

    matrix = evaluate_matrix(
        aperture,
        context={"wavelength": wavelength},
    )

    expected = np.array([
        [1.0, 0.0],
        [
            -1j * wavelength / (np.pi * radius**2),
            1.0,
        ],
    ])

    np.testing.assert_allclose(matrix, expected)