"""Boundary views use frozen element identity and preserve trace ordering."""

import numpy as np
import pytest

from diffractix.backend import xp
from diffractix.input import Ray, RayBundle
from diffractix.simulation.info import SimulationStep
from diffractix.simulation.result import SimulationResult, SimulationResultView
from diffractix.system.info import ElementData, ParameterData


def make_result(elements, *, positions=None, states=None):
    elements = tuple(elements)
    if states is None:
        states = tuple(Ray.numerical_type()(x=.001*index, theta=.01*index, n=1.) for index in range(len(elements)+1))
    data = tuple(ElementData(id(element), index, "Lens", f"Lens {index}", None, frozenset(), ("f",), (0,))
                 for index, element in enumerate(elements))
    return SimulationResult(
        theta=[], parameter_data=(), element_data=data,
        optical_stages=(SimulationStep(((0, 0), (0, 0)), 0, 0),)*len(elements),
        optical_values=(.1,), z=xp.array(range(len(states)) if positions is None else positions), states=states,
    )


def test_stage_views_align_states_positions_and_metadata():
    first, second = object(), object()
    result = make_result((first, second), positions=[0., .1, .1])
    assert result.source is result.initial is result.states[0]
    assert result.final is result.states[2]
    assert len(result.stages) == 2
    for index, element in enumerate((first, second)):
        view = result[element]
        assert view.before is result.states[index]
        assert view.after is result.states[index+1]
        assert result.stage(element).before is result.stages[index].before is view.before
        assert view.z_start == result.z[index]
        assert view.z_end == result.z[index+1]
        info = result.element_data[index]
        assert info.element_id == id(element)
        assert dict(zip(info.parameter_names, (result.optical_values[i] for i in info.parameter_indices))) == {"f": .1}


def test_repeated_element_occurrences_select_distinct_boundary_pairs():
    repeated, middle = object(), object()
    result = make_result((repeated, middle, repeated), positions=[0., .1, .1, .4])
    for occurrence, index in ((0, 0), (1, 2)):
        view = result[repeated, occurrence]
        assert view.before is result.states[index]
        assert view.after is result.states[index+1]
        assert result.stage(repeated, occurrence).after is view.after
        assert view.z_start == result.z[index]
        assert view.z_end == result.z[index+1]


def test_result_view_exposes_stages_and_preserves_lookup_errors():
    repeated = object()
    result = make_result((repeated, repeated), positions=[0., .1, .2])
    view = result.view()
    assert isinstance(view, SimulationResultView)
    assert view.initial is result.initial
    assert view.final is result.final
    assert view[repeated, 1].before is result.states[1]
    assert view[repeated, 1].after is result.final
    assert view[repeated, 1].z_start == pytest.approx(.1)
    assert view[repeated, 1].z_end == pytest.approx(.2)
    with pytest.raises(ValueError, match="occurs"):
        view[repeated]
    with pytest.raises(IndexError, match="occurrence"):
        view[repeated, 2]
    with pytest.raises(KeyError, match="not part"):
        view[object()]
    with pytest.raises(AttributeError):
        view.plot()
    with pytest.raises(AttributeError):
        view.initial = result.final
    with pytest.raises(AttributeError):
        view._result = result


def test_result_view_preserves_derivatives_and_jit():
    from diffractix.backend import grad, jit

    element = object()

    def evaluate(x):
        states = (Ray.numerical_type()(x=x, theta=0., n=1.),
                  Ray.numerical_type()(x=x**2, theta=0., n=1.))
        return make_result((element,), states=states).view()[element].after.x

    assert grad(evaluate)(2.0) == pytest.approx(4.0)
    assert jit(grad(evaluate))(2.0) == pytest.approx(4.0)


def test_ambiguous_or_unknown_element_has_useful_lookup_errors():
    repeated = object()
    result = make_result((repeated, repeated))
    with pytest.raises(ValueError, match="occurs 2 times.*Specify occurrence"):
        result.stage(repeated)
    with pytest.raises(KeyError, match="not part"):
        result.stage(object())


@pytest.mark.parametrize("occurrence,error", [
    (-1, IndexError), (2, IndexError), (True, TypeError), (1.5, TypeError), ("1", TypeError),
])
def test_invalid_occurrence_is_rejected(occurrence, error):
    element = object()
    result = make_result((element, element))
    with pytest.raises(error, match="occurrence"):
        result.stage(element, occurrence)


def test_numpy_integer_occurrence_and_tuple_lookup_work_but_numerical_index_does_not():
    element = object()
    result = make_result((element, element))
    assert result[element, np.int64(1)].after is result.final
    assert result.stages[0].before is result.initial
    with pytest.raises(KeyError, match="not part"):
        result[0]


@pytest.mark.parametrize("key", [(), (object(),), (object(), 0, "extra")])
def test_malformed_subscript_tuple_is_rejected(key):
    result = make_result(())
    with pytest.raises(TypeError, match="Expected result"):
        result[key]


@pytest.mark.parametrize("change,message", [
    ({"z": []}, "same length"),
    ({"states": [], "z": []}, "at least one state"),
    ({"element_data": []}, "one input state"),
    ({"optical_stages": []}, "optical_stages"),
    ({"theta": [1.]}, "theta and parameter_data"),
])
def test_constructor_rejects_inconsistent_trace_metadata(change, message):
    source = make_result((object(),))
    arguments = {name: getattr(source, name) for name in (
        "theta", "parameter_data", "element_data", "optical_stages", "optical_values", "z", "states",
    )}
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        SimulationResult(**arguments)


def test_result_snapshots_sequence_containers_and_theta_not_live_parameter_values():
    source = make_result((object(),))
    theta = np.array([.2])
    parameters = [ParameterData(123, "f", .1, 0., 1., 0)]
    elements, stages, states, values = list(source.element_data), list(source.optical_stages), list(source.states), [.1]
    result = SimulationResult(theta=theta, parameter_data=parameters, element_data=elements,
                              optical_stages=stages, optical_values=values, z=source.z, states=states)
    theta[:] = .9
    for sequence in (parameters, elements, stages, states, values):
        sequence.clear()
    np.testing.assert_array_equal(result.theta, [.2])
    assert result.parameter_data[0].value == .1
    assert len(result.element_data) == len(result.optical_stages) == len(result.optical_values) == 1
    assert len(result.states) == 2


@pytest.mark.parametrize("state_kind", ["ray", "bundle"])
def test_string_result_reports_boundary_locations_and_state_values(state_kind):
    element = object()
    if state_kind == "bundle":
        states = tuple(RayBundle.numerical_type()(x=xp.array([.001, .002])+offset,
                        theta=xp.array([.01, .02]), n=1.) for offset in (0., .001))
    else:
        states = None
    result = make_result((element,), states=states)
    text = str(result)
    assert "Simulation Result" in text and "Input" in text and "Lens 0" in text
    assert "x" in text and "theta" in text
    assert "States: 2" in text
    assert result[element].after is result.final
