"""Fake optical results and canonical parameter snapshots for solver tests."""

from dataclasses import dataclass
from types import SimpleNamespace
import autograd.numpy as np
from diffractix.graph import Parameter
from diffractix.simulation import Simulation
from diffractix.solver.solver import Solver, SolverContext


@dataclass
class FakeState:
    w: object
    R: object
    q: object


class FakeResult:
    def __init__(self, theta, element):
        self.theta = theta
        self.element = element
        self.z = np.stack([0.0 * theta[0], theta[0] + 10.0])
        self.states = (
            FakeState(theta[0] + 1.0, theta[0] + 2.0, theta[0] + 3.0),
            FakeState(theta[0] + 2.0, theta[0] + 3.0, theta[0] + 4.0),
        )

    def at(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return FakeState(
            self.theta[0] + 1.0 + occurrence,
            self.theta[0] + 2.0 + occurrence,
            self.theta[0] + 3.0 + occurrence,
        )

    def after(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return FakeState(
            self.theta[0] + 2.0 + occurrence,
            self.theta[0] + 3.0 + occurrence,
            self.theta[0] + 4.0 + occurrence,
        )

    def z_at(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return self.theta[0] + 20.0 + occurrence

    def z_after(self, element, occurrence=None):
        if element is not self.element:
            raise KeyError("Unknown element in fake result.")
        occurrence = 0 if occurrence is None else occurrence
        return self.theta[0] + 30.0 + occurrence


class FakeElement:
    pass


def make_solver(value=2.0):
    element = FakeElement()
    parameter = Parameter(value, name="p").variable()
    element.p = parameter

    simulation = object.__new__(Simulation)
    simulation.parameter_info = {
        id(parameter): SimpleNamespace(
            value=value,
            parameter_index=0,
            lower_bound=-10.0,
            upper_bound=10.0,
        ),
    }
    simulation.compile_context = {}
    simulation.execution_context = {}
    simulation.requirements = ()
    simulation.run = lambda theta: FakeResult(theta, element)

    return Solver(simulation), element, parameter


def context(solver, theta):
    return SolverContext(np.array(theta), solver.simulation.run)


def make_solver_with_reordered_parameters():
    solver, element, first = make_solver()
    second = Parameter(4.0, name="second").variable()

    solver.simulation.parameter_info = {
        id(first): SimpleNamespace(
            value=2.0,
            parameter_index=1,
            lower_bound=-2.0,
            upper_bound=2.0,
        ),
        id(second): SimpleNamespace(
            value=4.0,
            parameter_index=0,
            lower_bound=-4.0,
            upper_bound=4.0,
        ),
    }

    return solver, element, first, second
