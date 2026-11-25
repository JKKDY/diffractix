"""Diffractix: differentiable paraxial optics and inverse design."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("diffractix")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

# Optical inputs
from .core import OpticalBase
from . import units
from .states import (
    OpticalState,
    ParaxialState,
    GaussianBeam,
    Ray,
    RayBundle,
)

# Elements
from .elements import (
    OpticalElement,
    parameter,
    Space,
    ThinLens,
    Mirror,
    Interface,
    ABCD,
    GaussianAperture,
    Plane,
    GRIN,
)

# Composites
from .composites import (
    CompositeElement,
    FourF,
    Slab,
    ThickLens,
    Telescope,
)

# Symbolic graph
from .graph import (
    Node,
    Parameter,
    ComplexParameter,
    Symbol,
    SymbolicControlFlowError,
)

# Systems and simulation
from .system import (
    System,
    SystemValidationError,
    AMBIENT_N,
    WAVELENGTH,
)
from .simulation import (
    Simulation,
    SimulationResult,
)
from .simulation.result import SimulationResultView

# Inverse design
from .solver import (
    Solver,
    Optimizer,
    Objective,
    Constraint,
    Solution,
)


__all__ = [
    # Optical inputs
    "OpticalBase",
    "units",
    "OpticalState",
    "ParaxialState",
    "GaussianBeam",
    "Ray",
    "RayBundle",

    # Elements
    "OpticalElement",
    "parameter",
    "Space",
    "ThinLens",
    "Mirror",
    "Interface",
    "ABCD",
    "GaussianAperture",
    "Plane",
    "GRIN",

    # Composites
    "CompositeElement",
    "FourF",
    "Slab",
    "ThickLens",
    "Telescope",

    # Graph
    "Node",
    "Parameter",
    "ComplexParameter",
    "Symbol",
    "SymbolicControlFlowError",

    # Systems and simulation
    "System",
    "SystemValidationError",
    "AMBIENT_N",
    "WAVELENGTH",
    "Simulation",
    "SimulationResult",
    "SimulationResultView",

    # Inverse design
    "Solver",
    "Optimizer",
    "Objective",
    "Constraint",
    "Solution",
]
