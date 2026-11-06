"""Editable guesses address canonical independent Parameters by identity."""

import numpy as np
import pytest

from diffractix.graph import InputNode, Parameter
from diffractix.simulation.simulation import InitialValues


def test_guess_copies_values_and_mapping_and_supports_reordered_direct_handles():
    first, second = Parameter(1.), Parameter(2.)
    values = np.array([2., 1.])
    indices = {id(second): 0, id(first): 1}
    guess = InitialValues(values, indices)
    values[:] = 99.
    indices.clear()

    guess[InputNode(first)] = 3.
    assert guess[first] == 3. and guess[second] == 2.
    assert first.value == 1. and second.value == 2.
    np.testing.assert_array_equal(guess.values, [2., 3.])


@pytest.mark.parametrize("key_factory", [lambda p: p, lambda p: InputNode(p)])
def test_missing_parameter_identity_does_not_match_equal_metadata(key_factory):
    first, unrelated = Parameter(1., name="same"), Parameter(1., name="same")
    guess = InitialValues([1.], {id(first): 0})
    with pytest.raises(KeyError, match="not part"):
        guess[key_factory(unrelated)] = 2.
    np.testing.assert_array_equal(guess.values, [1.])


@pytest.mark.parametrize("key_factory", [
    lambda p: p+1, lambda p: InputNode(p+1), lambda p: InputNode(InputNode(p)),
    lambda p: InputNode(None), lambda p: "p", lambda p: 0,
])
def test_guess_rejects_expression_indirect_empty_and_nonparameter_keys(key_factory):
    parameter = Parameter(1.)
    guess = InitialValues([1.], {id(parameter): 0})
    with pytest.raises(TypeError, match="Parameter"):
        guess[key_factory(parameter)] = 2.
    np.testing.assert_array_equal(guess.values, [1.])


@pytest.mark.parametrize("value,error", [
    (True, TypeError), (1j, TypeError), ("2", TypeError), ([2.], TypeError),
    (object(), TypeError), (np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError),
])
def test_invalid_assignment_preserves_previous_guess(value, error):
    parameter = Parameter(1.)
    guess = InitialValues([1.], {id(parameter): 0})
    with pytest.raises(error):
        guess[parameter] = value
    assert guess[parameter] == 1.


def test_assignment_uses_frozen_membership_and_allows_later_bound_overrides():
    parameter = Parameter(1.).variable(bounds=(0., 2.))
    guess = InitialValues([1.], {id(parameter): 0})
    parameter.fixed()
    guess[parameter] = np.float32(3.)
    assert guess[parameter] == 3.
    assert parameter.value == 1. and not parameter.is_variable


def test_empty_theta_has_an_editable_empty_vector():
    guess = InitialValues([], {})
    assert guess.values.shape == (0,) and guess.values.flags.writeable
    with pytest.raises(KeyError, match="not part"):
        guess[Parameter(1.)]
