import autograd.numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from autograd import jacobian
from typing import Tuple
from .beams import GaussianBeam


@dataclass
class SimulationResult:
    # List of (position, beam) tuples
    trace: List[Tuple[float, GaussianBeam]]

    @property
    def final_beam(self) -> GaussianBeam:
        """Helper to get the output beam quickly."""
        return self.trace[-1][1]

    def plot_data(self):
        """Helper for plotting libraries (matplotlib/plotly)."""
        z_vals = [z for (t, g) in self.trace]
        w_vals = [g.w0 for (z, g) in self.trace] 
        return z_vals, w_vals

    def export(self) -> Dict[str, Any]:
        """
        Serializes for the UI.
        We perform the extraction of w and R here.
        """
        # Unzip the list of tuples
        zs = []
        ws = []
        Rs = []
        wavelength = self.trace[0][1].wavelength

        for z, beam in self.trace:
            zs.append(float(z))
            # Calculate physical properties for visualization
            # Note: We calculate w at the plane (offset=0)
            ws.append(float(beam.w_at_z(0))) 
            Rs.append(float(beam.R_at_z(0)))

        return {
            "structure": None,
            "results": [{
                "wavelength": wavelength,
                "z": zs,
                "w": ws,
                "R": Rs
            }]
        }




@dataclass(frozen=True)
class SimulationStep: 
    compute_matrix: callable   # function returning ABCD matrices
    param_indices: tuple[int]  
    length_index: int   
    index_index: int    


class Simulation:
    """
    The compiled, executable physics engine.
    This class is 'stateless' regarding the physics variables. 
    It defines the TOPOLOGY of the system.
    """
    def __init__(self, 
                steps: list[SimulationStep], 
                sources: list['GaussianBeam'],
                environment: "Environment",
                initial_values: np.array,
                parameter_transform: callable,
                constraints: list[callable] = None):
        assert len(sources) == 1, "Currently only single input beam supported."
                        
        # Physics
        self.steps = steps              
        self.initial_values = np.array(initial_values, dtype=float)
        self.sources = sources[0] 
        self.environment = environment

        # Metadata & Optimization Hooks
        self.parameter_transform = parameter_transform
        self.constraints = constraints if constraints is not None else []


    def _run(self, params:np.ndarray ) -> List[Tuple[float, complex]]:
        """
        Core physics loop:
        1. Expands optimization vars (params) to system state vector
        2. Propagates the beam using ABCD matrices.
        """
        
        if params is None:
            params = self.initial_values
        
        # explode from canconical set to full parameter set
        system_state = self.parameter_transform(params)
        wavelength = self.sources.wavelength
        
        history = []

        # initialize loop state
        q = self.sources.q
        z = 0.0 
        history = [(z, q, self.sources.n)]
        
        # Propagation Loop
        for step in self.steps:
            mat_params = system_state[step.param_indices]
            
            L = system_state[step.length_index]
            n = system_state[step.index_index]
            M = step.compute_matrix(*mat_params)
          
        
            # propagate with möbius transform
            A, B = M[0, 0], M[0, 1]
            C, D = M[1, 0], M[1, 1]

            numerator = A * q + B
            denominator = C * q + D
                     
            q = numerator / denominator
            z += L

            # record state
            history.append((z, q, n))

        return history

    def run_for_optimizer(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Math-Facing Forward Pass.
        Returns: A numpy array of shape (N_steps, 3) -> [z, w, R]
        """
        raw_history = self._run(params)
        wavelength = self.sources.wavelength
        
        # Convert to a flat array for efficient slicing in loss functions
        data = []
        for z, q, n in raw_history:
            w, R = GaussianBeam.w_R_from_q(q, wavelength, n)
            data.append([z, w, R])
            
        return np.stack(data) # Shape: (Steps, 3)


    def run(self, params: np.ndarray | None = None) -> 'SimulationResult':
        """
        User-Facing Forward Pass.
        Returns: Rich objects for the UI/Scripting.
        """
        raw_history = self._run(params)
        wavelength = self.sources.wavelength
        
        # Convert raw (z, q) -> (z, GaussianBeam) for the Result object
        trace = [(z, GaussianBeam(q, wavelength, n)) for z, q, n in raw_history]

        return SimulationResult(trace)

