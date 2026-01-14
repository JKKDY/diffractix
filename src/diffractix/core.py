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
 



def w_R_from_q(q, wavelength, n=1.0):
    """
    Extracts physical properties (waist radius w, curvature R) from the complex q parameter.
    
    Formulas:
    1/q = 1/R - i * (lambda / (pi * n * w^2))
    
    Args:
        q (complex): The complex beam parameter.
        wavelength (float): The wavelength in vacuum (meters).
        n (float): The refractive index of the medium (default 1.0).
        
    Returns:
        (w, R): Tuple of floats (spot size radius, radius of curvature).
    """
    inv_q = 1.0 / q
    
    # Curvature R = 1 / Re(1/q)
    # Handle the plane wave case (Re(1/q) == 0) carefully if needed, 
    # but normally numpy handles inf correctly.
    # We add a tiny epsilon to avoid div-by-zero if strictly 0.
    real_part = np.real(inv_q)
    R = 1.0 / (real_part + 1e-20) 
    
    # Spot Size w
    # Im(1/q) = - lambda / (pi * n * w^2)
    # w^2 = - lambda / (pi * n * Im(1/q))
    imag_part = np.imag(inv_q)
    w_squared = -wavelength / (np.pi * n * imag_part)
    w = np.sqrt(w_squared)
    
    return w, R