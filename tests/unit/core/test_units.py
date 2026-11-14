"""Unit constants are plain SI scale factors with a symbol for formatting."""

import math

import numpy as np
import pytest

import diffractix
from diffractix import GaussianBeam, Parameter, ThinLens, units
from diffractix.core.units import Unit, deg, m, mm, mrad, nm, um


def test_units_are_si_scale_factors_and_arithmetic_gives_plain_floats():
    assert (m, mm, um, nm) == (1.0, 1e-3, 1e-6, 1e-9)
    assert deg == pytest.approx(math.pi / 180) and mrad == 1e-3
    for value in (50 * mm, mm * 50, 3 / mm, mm + um, -mm):
        assert type(value) is float
    assert (50 * mm) / mm == pytest.approx(50)
    assert type(np.array([1.0, 2.0]) * mm) is np.ndarray


def test_units_flow_into_elements_parameters_and_inputs_as_si_values():
    lens = ThinLens(f=50 * mm).variable(f=(20 * mm, 100 * mm))
    assert lens.f.value == pytest.approx(0.05)
    assert lens.f.node.bounds == pytest.approx((0.02, 0.1))
    assert Parameter(mm).value == 1e-3
    beam = GaussianBeam.from_waist(w0=500 * um, wavelength=1064 * nm).numeric()
    assert beam.w == pytest.approx(5e-4)


@pytest.mark.parametrize("value, unit, digits, expected", [
    (4.5e-5, um, 4, "45 um"),
    (4.50213e-5, um, 4, "45.02 um"),
    (0.0123456, mm, 3, "12.3 mm"),
    (0.0, mm, 4, "0 mm"),
    (np.array([1e-3, 2.5e-3]), mm, 4, "[1, 2.5] mm"),
    (math.pi / 2, deg, 4, "90 deg"),
])
def test_format_writes_a_value_in_the_given_unit(value, unit, digits, expected):
    assert units.format(value, unit, digits) == expected


@pytest.mark.parametrize("unit, digits, error", [
    (1e-3, 4, TypeError), ("mm", 4, TypeError), (mm, 0, ValueError), (mm, True, ValueError),
])
def test_format_rejects_plain_numbers_as_units_and_invalid_digits(unit, digits, error):
    with pytest.raises(error):
        units.format(1.0, unit, digits)


def test_units_module_is_reachable_but_does_not_flood_the_top_level_namespace():
    from diffractix.units import mm as public_mm, format as public_format
    from diffractix.core import units as core_units
    assert public_mm is mm and public_format is core_units.format
    assert units.__all__ == core_units.__all__
    assert diffractix.units is units
    assert "units" in diffractix.__all__
    assert not {"m", "mm", "um", "nm", "deg"} & set(diffractix.__all__)
    assert "format" not in units.__all__
    assert isinstance(mm, Unit) and repr(mm) == "Unit(0.001, 'mm')"
