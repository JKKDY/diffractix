"""Concrete array literals own their data and preserve numerical metadata."""

import numpy as np
import pytest

from diffractix.backend import grad, xp
from diffractix.graph import Literal, Node, Parameter


@pytest.mark.parametrize("factory", [Literal, Node._make_literal])
@pytest.mark.parametrize("value", [
    np.array([1, 2], dtype=np.int16),
    np.array([1, 2], dtype=np.uint32),
    np.array([[1, 2], [3, 4]], dtype=np.float32),
    np.array([1 + 2j, -0j], dtype=np.complex64),
    np.array([True, False]),
    np.empty((0, 2), dtype=np.float64),
    np.arange(12, dtype=np.float64).reshape(3, 4)[::-1, ::2],
])
def test_array_literals_preserve_shape_dtype_and_own_read_only_data(factory, value):
    literal = factory(value)
    np.testing.assert_array_equal(literal.value, value)
    assert literal.value.shape == value.shape
    assert literal.value.dtype == value.dtype
    assert literal.value.flags.owndata
    assert not literal.value.flags.writeable
    assert not np.shares_memory(literal.value, value)
    assert not literal.is_variable
    with pytest.raises(AttributeError):
        literal.value = value


@pytest.mark.parametrize("factory", [Literal, Node._make_literal])
def test_mutating_source_array_or_view_cannot_change_literal(factory):
    original = np.arange(6, dtype=np.float32)
    literal = factory(original[::2])
    original[:] = 100
    np.testing.assert_array_equal(literal.value, [0, 2, 4])
    with pytest.raises(ValueError, match="read-only"):
        literal.value[0] = 9


@pytest.mark.parametrize("value", [
    np.array(["text"]), np.array([b"bytes"]), np.array([object()], dtype=object),
    np.array([Parameter(1)], dtype=object),
    np.array(["2026-01-01"], dtype="datetime64[D]"),
    np.array([1], dtype="timedelta64[D]"),
    np.array([(1,)], dtype=[("field", "i4")]),
])
@pytest.mark.parametrize("factory", [Literal, Node._make_literal])
def test_arrays_with_unsupported_dtypes_are_rejected(value, factory):
    with pytest.raises(TypeError, match="numeric dtype"):
        factory(value)


@pytest.mark.parametrize("value", [[1, 2], (1, 2), None, "text"])
def test_array_literal_construction_requires_explicit_numerical_arrays(value):
    with pytest.raises(TypeError, match="numeric scalar or array"):
        Literal(value)


def test_backend_array_literal_uses_owned_numpy_storage():
    value = xp.array([1, 2], dtype=xp.float32)
    literal = Node._make_literal(value)
    assert isinstance(literal.value, np.ndarray)
    assert literal.value.dtype == np.dtype("float32")
    np.testing.assert_array_equal(literal.value, [1, 2])
    assert not literal.value.flags.writeable


@pytest.mark.parametrize("factory", [Literal, Node._make_literal])
def test_scalar_array_normalization_is_preserved(factory):
    value = np.array(2.0)
    literal = factory(value)
    value[...] = 99
    assert isinstance(literal.value, float)
    assert literal.value == 2


@pytest.mark.parametrize("factory", [Literal, Node._make_literal])
def test_active_differentiation_values_cannot_become_literals(factory):
    def function(value):
        literal = factory(value)
        return xp.sum(literal.value)

    with pytest.raises(TypeError, match="numeric scalar or array"):
        grad(function)(xp.array([1., 2.]))


def test_parameters_remain_scalar_valued():
    with pytest.raises(TypeError, match="real scalar"):
        Parameter(np.array([1., 2.]))
