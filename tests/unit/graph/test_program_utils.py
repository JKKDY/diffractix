"""Canonical parameter/symbol alignment and program fusion preserve mathematics."""
import numpy as np
import pytest

from diffractix.graph import (
    Literal, Parameter, Symbol, compile_graph, fuse, precompile,
    remap_parameters, remap_runtime_symbols,
)


def test_parameter_remapping_reorders_and_extends_theta_including_direct_outputs():
    first, second, unused = (Parameter(value).variable() for value in (1, 2, 3))
    source = precompile((first + 2*second + Symbol("offset"), -first, second, Literal(9)))
    original_mapping = dict(source.parameter_indices)
    canonical = {id(unused): 0, id(second): 1, id(first): 2}
    remapped = remap_parameters(source, canonical)
    assert remapped.parameter_indices == canonical
    assert source.parameter_indices == original_mapping
    assert source.runtime_symbols == remapped.runtime_symbols == {"offset": 0}
    np.testing.assert_allclose(remapped.evaluate([99, 5, 3], {"offset": 4}), [17, -3, 5, 9])
    np.testing.assert_allclose(source.evaluate([3, 5], {"offset": 4}), [17, -3, 5, 9])


def test_parameter_remapping_rejects_missing_required_parameter():
    parameter = Parameter(1.0).variable()
    source = compile_graph((parameter,))
    with pytest.raises(ValueError, match="Missing required Parameter"):
        remap_parameters(source, {999: 0})


@pytest.mark.parametrize("mapping", [{123: 1}, {123: 0, 456: 0}, {123: -1}])
def test_parameter_remapping_rejects_nonconsecutive_canonical_indices(mapping):
    source = compile_graph((Literal(1),))
    with pytest.raises(ValueError, match="consecutive"):
        remap_parameters(source, mapping)


def test_runtime_symbol_remapping_handles_reordered_slots_direct_outputs_and_theta():
    parameter = Parameter(1.0).variable()
    source = compile_graph((Symbol("a") - Symbol("b") + parameter, Symbol("b"), Literal(2)))
    mapping = {"unused": 0, "b": 1, "a": 2}
    remapped = remap_runtime_symbols(source, mapping)
    assert source.runtime_symbols == {"a": 0, "b": 1}
    assert remapped.runtime_symbols == mapping
    assert remapped.parameter_indices == source.parameter_indices
    np.testing.assert_allclose(remapped.evaluate([2], {"a": 7, "b": 3, "unused": 99}), [6, 3, 2])
    np.testing.assert_allclose(source.evaluate([2], {"a": 7, "b": 3}), [6, 3, 2])


def test_runtime_symbol_remapping_rejects_missing_required_binding():
    source = compile_graph((Symbol("a"),))
    with pytest.raises(ValueError, match="Missing required runtime symbols"):
        remap_runtime_symbols(source, {"b": 0})


def test_fusion_offsets_reused_registers_and_keeps_direct_and_duplicate_outputs():
    parameter = Parameter(1.0).variable()
    first = compile_graph((-(parameter + Symbol("s")), parameter, Literal(7)))
    second = compile_graph(((parameter + Symbol("s"))**2, Symbol("s")))
    first_before = (dict(first.parameter_indices), first._instructions, first._outputs)
    second_before = (dict(second.runtime_symbols), second._instructions, second._outputs)
    combined = fuse(first, second, first)
    np.testing.assert_allclose(combined.evaluate([2], {"s": 3}), [-5, 2, 7, 25, 3, -5, 2, 7])
    assert (first.parameter_indices, first._instructions, first._outputs) == first_before
    assert (second.runtime_symbols, second._instructions, second._outputs) == second_before


def test_fusion_accepts_empty_programs_but_requires_at_least_one_program():
    empty = compile_graph(())
    constant = compile_graph((Literal(3),))
    assert fuse(empty, constant, empty).evaluate([]) == (3,)
    with pytest.raises(ValueError, match="at least one program"):
        fuse()


@pytest.mark.parametrize("difference", ["parameter_identity", "parameter_order", "symbol_keys", "symbol_order"])
def test_fusion_rejects_incompatible_input_conventions(difference):
    x, y = Parameter(1.0).variable(), Parameter(1.0).variable()
    if difference == "parameter_identity":
        first, second = compile_graph((x,)), compile_graph((y,))
    elif difference == "parameter_order":
        first, second = compile_graph((x + y,)), compile_graph((y + x,))
    elif difference == "symbol_keys":
        first, second = compile_graph((Symbol("a"),)), compile_graph((Symbol("b"),))
    else:
        first = compile_graph((Symbol("a") + Symbol("b"),))
        second = compile_graph((Symbol("b") + Symbol("a"),))
    with pytest.raises(ValueError, match="identical"):
        fuse(first, second)
