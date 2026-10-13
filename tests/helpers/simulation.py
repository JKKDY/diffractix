"""Minimal Simulation snapshots for solver and solution tests."""

from __future__ import annotations

from types import SimpleNamespace
import autograd.numpy as np
from diffractix.simulation import Simulation
from diffractix.system.system import ParameterInfo


def parameter_info(parameter, index, *, value=None, lower=-np.inf, upper=np.inf):
    return ParameterInfo(
        parameter_id=id(parameter),
        name=parameter.name,
        value=parameter.value if value is None else value,
        parameter_index=index,
        lower_bound=lower,
        upper_bound=upper,
    )


def make_simulation(parameters, *, run=None):
    simulation = object.__new__(Simulation)
    simulation.parameter_info = {
        id(parameter): parameter_info(
            parameter,
            index,
            lower=parameter.lower_bound,
            upper=parameter.upper_bound,
        )
        for index, parameter in enumerate(parameters)
    }
    simulation.compile_context = {}
    simulation.execution_context = {}
    simulation.requirements = ()
    simulation.graph = SimpleNamespace(
        initial_values=np.array([parameter.value for parameter in parameters])
    )
    simulation.run = run or (lambda theta: SimpleNamespace(theta=theta))
    return simulation
