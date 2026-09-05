import pytest

from pathlib import Path
from dataclasses import dataclass

from diffractix.beams import ParaxialRay
from diffractix.composites import Slab
from diffractix.composites import CompositeElement
from diffractix.system import AMBIENT_N
from diffractix.elements import OpticalElement, ThinLens
from diffractix.graph import Node, Parameter
from diffractix.system.system import System, Placement, SourceInfo
from diffractix.system.errors import SystemValidationError



def make_source_info():
    return SourceInfo(
        file="test_system.py",
        line=100,
        call_index=0,
    )


def evaluate_graph(compiled, values=()):
    return np.asarray(
        compiled.graph.evaluate(np.asarray(values, dtype=float))
    )


@dataclass(kw_only=True)
class FailingElement(OpticalElement):

    x: Node

    @property
    def matrix(self):
        return (
            (1.0, self.x),
            (0.0, 1.0),
        )

    @property
    def element_length(self):
        return 0.0

    def _validate_for_build(self):
        raise ValueError("intentional validation failure")


class FailingComposite(CompositeElement):

    def __init__(self):
        self.good = ThinLens(f=0.1)
        self.bad = FailingElement(x=1.0)
        super().__init__()


def make_beam():
    return ParaxialRay(
        x=0.0,
        theta=0.0,
    )


# -----------
# SOURCE INFO
# -----------

def test_source_info_string():
    info = SourceInfo(
        file="example.py",
        line=42,
        call_index=0,
    )

    assert str(info) == "example.py:42"


def test_placement_description():
    lens = ThinLens(f=0.1, label="Lens")
    placement = Placement(
        element=lens,
        z=0.5,
        source_info=SourceInfo(
            file="example.py",
            line=42,
            call_index=0,
        ),
    )

    assert placement.describe(2) == (
        "Placement #2 (ThinLens 'Lens') added at example.py:42"
    )


def test_placement_description_without_index():
    lens = ThinLens(f=0.1, label="Lens")
    placement = Placement(element=lens)

    assert placement.describe() == "Placement (ThinLens 'Lens')"


def test_add_tracks_repeated_source_line_execution():
    system = System()

    for _ in range(3):
        system.add(ThinLens(f=0.1))

    assert [placement.source_info.call_index for placement in system.placements] == [
        0,
        1,
        2,
    ]

# ---------------------
# SYSTEM INITIALIZATION
# ---------------------

def test_system_initializes_ambient_index():
    system = System()

    assert system.ambient_n.value == pytest.approx(1.0)
    assert not system.ambient_n.is_variable


def test_system_ambient_index_can_be_variable():
    system = System(
        ambient_n=1.33,
        ambient_n_variable=True,
    )

    assert system.ambient_n.value == pytest.approx(1.33)
    assert system.ambient_n.is_variable


def test_ambient_index_is_registered_in_context():
    system = System()

    assert system.context[AMBIENT_N.name] is system.ambient_n


def test_system_starts_without_elements():
    system = System()

    assert system.elements == ()
    assert system.placements == ()


# -----------
# ADD ELEMENT
# -----------

def test_add_single_element():
    system = System()
    lens = ThinLens(f=0.1)

    result = system.add(lens)

    assert result is system
    assert system.elements == (lens,)
    assert len(system.placements) == 1
    assert system.placements[0].element is lens


def test_add_preserves_element_identity():
    system = System()
    lens = ThinLens(f=0.1)

    system.add(lens)

    assert system.elements[0] is lens


def test_add_fixed_absolute_position():
    system = System()
    lens = ThinLens(f=0.1)

    system.add(lens, z=0.5)

    assert system.placements[0].z == pytest.approx(0.5)


def test_add_parameter_absolute_position():
    system = System()
    lens = ThinLens(f=0.1)
    z = Parameter(0.5).variable()

    system.add(lens, z=z)

    assert system.placements[0].z is z
    assert z.is_variable


def test_add_expression_absolute_position():
    system = System()
    lens1 = ThinLens(f=0.1)
    lens2 = ThinLens(f=0.2)
    z = lens1.f + 0.5

    system.add(lens2, z=z)

    assert system.placements[0].z is z


def test_add_multiple_elements():
    system = System()
    first = ThinLens(f=0.1)
    second = ThinLens(f=0.2)

    system.add([first, second])

    assert system.elements == (
        first,
        second,
    )


def test_add_tuple_of_elements():
    system = System()
    first = ThinLens(f=0.1)
    second = ThinLens(f=0.2)

    system.add((first, second))

    assert system.elements == (
        first,
        second,
    )


def test_add_generator_of_elements():
    system = System()
    elements = [
        ThinLens(f=0.1),
        ThinLens(f=0.2),
        ThinLens(f=0.3),
    ]

    system.add(element for element in elements)

    assert system.elements == tuple(elements)


def test_add_composite_as_single_element():
    system = System()
    slab = Slab(d=0.01, n=1.5)

    system.add(slab)

    assert system.elements == (slab,)
    assert system.placements[0].element is slab


def test_add_sequence_with_absolute_position_is_rejected():
    system = System()
    elements = [
        ThinLens(f=0.1),
        ThinLens(f=0.2),
    ]

    with pytest.raises(ValueError):
        system.add(elements, z=0.5)


@pytest.mark.parametrize(
    "z",
    [
        True,
        False,
        "0.5",
        object(),
        [],
        {},
        1 + 2j,
    ],
)
def test_add_rejects_invalid_absolute_position_types(z):
    system = System()
    lens = ThinLens(f=0.1)

    with pytest.raises(TypeError):
        system.add(lens, z=z)


@pytest.mark.parametrize(
    "element",
    [
        None,
        1,
        1.5,
        True,
        object(),
        "lens",
        b"lens",
    ],
)
def test_add_rejects_invalid_element_types(element):
    system = System()

    with pytest.raises(TypeError):
        system.add(element)


def test_add_sequence_rejects_invalid_child():
    system = System()
    lens = ThinLens(f=0.1)

    with pytest.raises(TypeError):
        system.add([lens, object()])


def test_add_supports_chaining():
    system = System()
    first = ThinLens(f=0.1)
    second = ThinLens(f=0.2)

    result = system.add(first).add(second)

    assert result is system
    assert system.elements == (
        first,
        second,
    )


def test_add_records_source_information():
    system = System()
    lens = ThinLens(f=0.1)

    system.add(lens)

    info = system.placements[0].source_info

    assert info is not None
    assert Path(info.file).name == Path(__file__).name
    assert info.line > 0
    assert info.call_index == 0


# -------
# CONTEXT
# -------
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "293.15",
        object(),
        [],
        {},
        1 + 2j,
    ],
)
def test_add_context_rejects_invalid_values(value):
    system = System()

    with pytest.raises(TypeError):
        system.add_context("temperature", value)

def test_add_numeric_context_value():
    system = System()

    result = system.add_context("temperature", 293.15)

    assert result is system
    assert isinstance(system.context["temperature"], Parameter)
    assert system.context["temperature"].value == pytest.approx(293.15)
    assert not system.context["temperature"].is_variable


def test_add_context_preserves_node_identity():
    system = System()
    temperature = Parameter(293.15).variable()

    system.add_context("temperature", temperature)

    assert system.context["temperature"] is temperature


def test_add_context_replaces_existing_value():
    system = System()

    system.add_context("temperature", 293.15)
    system.add_context("temperature", 300.0)

    assert system.context["temperature"].value == pytest.approx(300.0)


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        123,
        False,
    ],
)
def test_add_context_rejects_invalid_names(name):
    system = System()

    with pytest.raises(ValueError):
        system.add_context(name, 1.0)


# ----
# BEAM
# ----

def test_add_input_beam():
    system = System()
    beam = make_beam()

    result = system.add_input_beam(beam)

    assert result is system
    assert system.beam is beam


def test_add_input_beam_replaces_previous_beam():
    system = System()
    first = make_beam()
    second = ParaxialRay(
        x=0.1,
        theta=0.01,
    )

    system.add_input_beam(first)
    system.add_input_beam(second)

    assert system.beam is second


# ---------------------
# SYSTEM VALIDATION
# ---------------------

def test_validate_accepts_valid_system():
    system = System()
    system.add_input_beam(make_beam())
    system.add(ThinLens(f=0.1))

    system._validate()


def test_validate_requires_input_beam():
    system = System()
    system.add(ThinLens(f=0.1))

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    assert "No input beam has been set" in str(exc_info.value)


def test_validate_rejects_invalid_input_beam():
    system = System()
    system.add_input_beam(object())

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    assert "Input beam must be a ParaxialState" in str(exc_info.value)


@pytest.mark.parametrize(
    "ambient_n",
    [
        0.0,
        -1.0,
        -1.33,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_validate_rejects_invalid_ambient_index(ambient_n):
    system = System(ambient_n=ambient_n)
    system.add_input_beam(make_beam())

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    assert "Ambient refractive index must be finite and positive" in str(exc_info.value)


@pytest.mark.parametrize(
    "z",
    [
        -1.0,
        -0.1,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_validate_rejects_invalid_fixed_absolute_positions(z):
    system = System()
    system.add_input_beam(make_beam())
    system.add(ThinLens(f=0.1), z=z)

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    assert "absolute position z" in str(exc_info.value)


def test_validate_calls_element_validation():
    system = System()
    system.add_input_beam(make_beam())
    system.add(FailingElement(x=1.0, label="Broken"))

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    text = str(exc_info.value)

    assert "Broken" in text
    assert "intentional validation failure" in text


def test_validate_calls_leaf_validation_for_composite():
    system = System()
    system.add_input_beam(make_beam())
    system.add(FailingComposite())

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    text = str(exc_info.value)

    assert "bad" in text
    assert "FailingElement" in text
    assert "intentional validation failure" in text



def test_validate_reports_placement_location():
    system = System()
    system.add_input_beam(make_beam())
    system.add(FailingElement(x=1.0, label="Broken"))

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    text = str(exc_info.value)

    assert "Placement #0" in text
    assert "Broken" in text
    assert Path(__file__).name in text


def test_validate_collects_multiple_errors():
    system = System(ambient_n=-1.0)
    system.add(ThinLens(f=0.1), z=-0.5)

    with pytest.raises(SystemValidationError) as exc_info:
        system._validate()

    error = exc_info.value

    assert len(error.errors) >= 3
    assert "No input beam has been set" in str(error)
    assert "Ambient refractive index" in str(error)
    assert "absolute position z" in str(error)


# ------------
# REQUIREMENTS
# ------------

def test_requirements_are_empty_by_default():
    system = System()

    assert system.requirements == ()


def test_require_adds_requirements():
    system = System()
    first = object()
    second = object()

    result = system.require(first, second)

    assert result is system
    assert system.requirements == (
        first,
        second,
    )


    