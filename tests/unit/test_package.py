"""Package-level exports and metadata."""

import diffractix
from diffractix.graph import SymbolicControlFlowError
from diffractix.simulation.result import SimulationResultView
from diffractix.system import AMBIENT_N, WAVELENGTH


def test_names_needed_for_custom_elements_and_callables_are_exported():
    assert diffractix.AMBIENT_N is AMBIENT_N and diffractix.WAVELENGTH is WAVELENGTH
    assert diffractix.SymbolicControlFlowError is SymbolicControlFlowError
    assert diffractix.SimulationResultView is SimulationResultView
    assert {"AMBIENT_N", "WAVELENGTH", "SymbolicControlFlowError", "SimulationResultView"} <= set(diffractix.__all__)
    assert all(hasattr(diffractix, name) for name in diffractix.__all__)


def test_version_is_a_string():
    assert isinstance(diffractix.__version__, str) and diffractix.__version__
