import uuid
import autograd.numpy as np
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class Entity:
    """Base class for all entities."""
    label: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metadata for UI sync (populated later by System.add)
    _source_info: dict = field(default_factory=dict, repr=False)
 



def w_R_from_q(q: complex, wavelength: float, n: float = 1.0) -> tuple[float, float]:
    """
    Extracts beam width (w) and radius of curvature (R) from the complex q-parameter.
    
    Math: 1/q = 1/R - i * (lambda / (pi * n * w^2))
    """
    inv_q = 1.0 / q
    
    # radius of Curvature
    if abs(inv_q.real) < 1e-16:
        R = np.inf
    else:
        R = 1.0 / inv_q.real
        
    # Beam Waist (corrected for index n)
    # Im(1/q) = - lambda / (pi * n * w^2)  =>  w = sqrt( -lambda / (pi * n * Im(1/q)) )
    imag_part = inv_q.imag
    
    # Safety: q.imag should always be negative for a valid Gaussian beam (forward propagation)
    # If positive, raise an error
    if imag_part > -1e-16: 
         raise Exception("Invalid q-parameter: Imaginary part of 1/q must be negative for a physical Gaussian beam.")

    w_sq = -wavelength / (np.pi * n * imag_part)
    w = np.sqrt(w_sq)
    
    return w, R