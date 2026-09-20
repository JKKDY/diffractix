"""Diffractix: differentiable paraxial optics and inverse design."""

# Beams
from .beams import (
    GaussianBeam,
    ParaxialRay,
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
    Symbol,
)

# Systems and simulation
from .system import (
    System,
    SystemValidationError,
)
from .simulation import (
    Simulation,
    SimulationResult,
)

# Inverse design
from .solver import (
    Solver,
    Backend,
    Objective,
    Constraint,
    Solution,
)


__all__ = [
    # Beams
    "GaussianBeam",
    "ParaxialRay",
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
    "Symbol",

    # Systems and simulation
    "System",
    "SystemValidationError",
    "Simulation",
    "SimulationResult",

    # Inverse design
    "Solver",
    "Backend",
    "Objective",
    "Constraint",
    "Solution",
]
