"""Layout, binding and requirement compilation across the public System API."""
import numpy as np
import pytest
from diffractix import Interface, Parameter, Ray, Slab, Space, System, ThinLens
from diffractix.composites import CompositeElement
from diffractix.graph import Symbol


class Relay(CompositeElement):
    def __init__(self,distance):
        self.window=Slab(d=distance,n=1.5)
        self.output=Space(d=distance)
        super().__init__()


def test_nested_composite_paths_shared_theta_and_absolute_gap():
    distance=Parameter(0.01).variable(bounds=(0.005,0.02))
    relay=Relay(distance)
    target=ThinLens(f=0.1,label="target")
    system=System(relay,optical_input=Ray(x=0.001,theta=0.01))
    system.add(target,z=0.05)
    simulation=system.build()
    result=simulation.run()
    assert len(simulation.parameter_data)==1
    assert [info.path for info in simulation.element_data[:4]]==["window.front","window.body","window.back","output"]
    assert result[target].z_start == pytest.approx(0.05)
    assert result.final.x == pytest.approx(0.001+0.01*0.01/1.5+0.04*0.01)
    guess=simulation.initial_values
    guess[distance]=0.02
    changed=simulation.run(guess.values)
    assert changed[target].z_start == pytest.approx(0.05)
    assert changed.final.x == pytest.approx(0.001+0.02*0.01/1.5+0.03*0.01)
    assert len(system.placements)==2
    assert len(simulation.constraint_descriptors)>=1  # variable generated-gap requirement


@pytest.mark.parametrize("attach", ["system","element"])
def test_unsatisfiable_fixed_constraints_fail_build(attach):
    element=Space(d=0.01)
    system=System(element,optical_input=Ray(x=0,theta=0))
    (system if attach=="system" else element).require(element.d>=0.02)
    with pytest.raises(ValueError,match="unsatisfiable"):
        system.build()


def test_binding_aliases_capture_shared_parameter_once():
    p=Parameter(0.02).variable()
    element=Space(d=Symbol("length"))
    system=System(element,optical_input=Ray(x=0.001,theta=0.01)).bind("length",2*p)
    simulation=system.build()
    assert len(simulation.parameter_data)==1
    assert simulation.run().z[-1] == pytest.approx(0.04)
    p.value=0.08
    assert simulation.run().z[-1] == pytest.approx(0.04)
    assert system.build().run().z[-1] == pytest.approx(0.16)


@pytest.mark.parametrize("case", ["missing-input","unresolved","external-variable","fixed-medium-mismatch"])
def test_invalid_models_fail_before_propagation(case):
    if case=="missing-input":
        system=System(Space(d=0.01))
    elif case=="unresolved":
        system=System(Space(d=Symbol("missing")),optical_input=Ray(x=0,theta=0))
    elif case=="external-variable":
        system=System(Space(d=0.01),optical_input=Ray(x=0,theta=0))
        system.require(Parameter(0.1).variable()>=0)
    else:
        system=System(Interface(n1=1.5,n2=1),optical_input=Ray(x=0,theta=0,n=1))
    with pytest.raises((ValueError,KeyError)):
        system.build()



def test_requirement_order_survives_nested_and_repeated_components_without_mutation():
    from diffractix import Constraint
    p=Parameter(0.01).variable()
    relay=Relay(p)
    system=System(relay,relay,optical_input=Ray(x=0,theta=0))
    system.require(Constraint(p,upper_bound=0.1,label="system"))
    relay.require(Constraint(p,lower_bound=0.001,label="composite"))
    relay.window.body.require(Constraint(p,upper_bound=0.02,label="child"))
    original=(system.system_constraints,relay.constraints,relay.window.body.constraints)
    simulation=system.build()
    named=[entry.label for entry in simulation.constraint_descriptors if entry.label]
    assert named[:3]==["system","composite","child"]
    assert named.count("child")==named.count("composite")==1
    assert len(simulation.parameter_data)==1
    assert len(simulation.element_data)==8
    assert (system.system_constraints,relay.constraints,relay.window.body.constraints)==original
    second=system.build()
    assert len(second.constraint_descriptors)==len(simulation.constraint_descriptors)


@pytest.mark.parametrize("variable",[False,True])
def test_negative_generated_gap_is_fixed_infeasibility_or_optimizable_constraint(variable):
    upstream=Parameter(0.01)
    if variable:
        upstream.variable(bounds=(0,0.02))
    system=System(Space(d=upstream),optical_input=Ray(x=0,theta=0.01))
    system.add(ThinLens(f=0.1),z=0.005)
    if not variable:
        with pytest.raises(ValueError,match="unsatisfiable"):
            system.build()
        return
    simulation=system.build()
    descriptors=simulation.constraint_descriptors
    outputs=simulation.constraint_program.evaluate(simulation.initial_values.values)
    assert any(value<entry.lower_bound for value,entry in zip(outputs,descriptors))
    guess=simulation.initial_values
    guess[upstream]=0.002
    values=simulation.constraint_program.evaluate(guess.values)
    assert all(entry.lower_bound<=value<=entry.upper_bound for value,entry in zip(values,descriptors))


def test_element_inspection_parameters_are_evaluated_expressions_not_live_fields():
    p=Parameter(0.02).variable()
    element=Space(d=2*p)
    simulation=System(element,optical_input=Ray(x=0,theta=0)).build()
    result=simulation.run([0.03])
    info=result.element_data[0]
    parameters=dict(zip(info.parameter_names,[result.optical_values[i] for i in info.parameter_indices]))
    assert parameters["d"]==pytest.approx(0.06)
    assert "n" not in parameters
    element.d=0.9
    assert simulation.run([0.03]).z[-1]==pytest.approx(0.06)
