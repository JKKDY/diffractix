"""Compiled propagation, trace lookup and custom component integration."""
from dataclasses import dataclass
import numpy as np
import pytest
from diffractix import Parameter, Ray, Space, System, ThinLens
from diffractix.elements import OpticalElement
from diffractix.graph import Node
from diffractix.input.paraxial_input import ParaxialState
from diffractix.backend import xp


def test_repeated_component_trace_and_occurrence_errors():
    distance = Parameter(0.02).variable()
    repeated = Space(d=distance)
    lens = ThinLens(f=0.1)
    simulation = System(repeated, lens, repeated, optical_input=Ray(x=0.001, theta=0.01)).build()
    guess = simulation.initial_values
    guess[distance] = 0.03
    result = simulation.run(guess.values)
    np.testing.assert_allclose(result.z, [0,0.03,0.03,0.06])
    assert result[repeated,0].before is result.initial
    assert result[repeated,1].after is result.final
    assert result[repeated,1].z_start == pytest.approx(0.03)
    assert result[repeated,1].z_end == pytest.approx(0.06)
    assert len(simulation.parameter_data) == 1
    with pytest.raises(ValueError, match="occurs"):
        result[repeated]
    with pytest.raises(KeyError):
        result[Space(d=0.02)]
    with pytest.raises(IndexError):
        result[repeated,2]
    with pytest.raises(TypeError):
        result[repeated,True]
    assert simulation.initial_values[distance] == 0.02


@dataclass
class CustomState(ParaxialState):
    height: Node
    slope: Node
    n: Node = 1.0
    @property
    def refractive_index(self):
        return self.n
    def propagate(self,A,B,C,D,n):
        return type(self)(height=A*self.height+B*self.slope,
                          slope=C*self.height+D*self.slope,n=n)


@dataclass(eq=False,kw_only=True)
class CustomTranslation(OpticalElement):
    distance: Node
    @property
    def matrix(self):
        return ((1,self.distance),(0,1))
    @property
    def element_length(self):
        return self.distance


def test_custom_state_and_element_use_same_compilation_pipeline():
    parameter=Parameter(0.2).variable()
    source=CustomState(height=0.001,slope=0.01)
    simulation=System(CustomTranslation(distance=parameter),optical_input=source).build()
    result=simulation.run(xp.array([0.3]))
    assert isinstance(result.final,CustomState)
    assert type(result.final) is CustomState.numerical_type()
    assert result.final.height == pytest.approx(0.004)
    assert result.final.slope == pytest.approx(0.01)
    assert result.z[-1] == pytest.approx(0.3)
    assert source.height.evaluate() == 0.001


def test_empty_path_and_invalid_theta_do_not_corrupt_next_run():
    p=Parameter(0.001).variable()
    simulation=System(optical_input=Ray(x=p,theta=0.01)).build()
    with pytest.raises(ValueError):
        simulation.run([])
    result=simulation.run([0.002])
    assert result.initial is result.final
    assert result.stages == ()
    np.testing.assert_array_equal(result.z,[0])
    assert simulation.run().final.x == pytest.approx(0.001)
