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
        z_vals = [t[0] for t in self.trace]
        # We assume n=1 for basic plotting, or could track n in history if needed
        w_vals = [t[1].w0(n=1.0) for t in self.trace] 
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
                initial_values: np.array,
                parameter_transform: callable,
                constraints: list[callable] = None):
        assert len(sources) == 1, "Currently only single input beam supported."
                        
        # Physics
        self.steps = steps              
        self.initial_values = np.array(initial_values, dtype=float)
        self.sources = sources[0] 

        # Metadata & Optimization Hooks
        self.parameter_transform = parameter_transform
        self.constraints = constraints if constraints is not None else []


    def _run(self, params:np.ndarray ) -> List[Tuple[float, complex]]:
        if params is None:
            params = self.initial_values
        
        # explode from canconical set to full parameter set
        params = self.parameter_transform(params)
        
        all_beams_history = []

        # initialize State
        q = self.sources.q
        z = 0.0  # starting at z=0
        wavelength = self.sources.wavelength
        current_n = self.environment.ambient_n.value
        
        # Initialize history with starting beam
        result = [(z, q, current_n)]

        # Propagation Loop
        for step in self.steps:
            step_params = params[list(step.param_indices)]

            # get current element properties
            mat = step.matrix_func(*step_params)
            step_z = step.length_func(*step_params)
            current_n = step.index_func(current_n, *step_params)
        
            # propagate with möbius transform
            A, B = mat[0]
            C, D = mat[1]
            numerator = A * q + B
            denominator = C * q + D
                     
            q = numerator / denominator
            z += step_z

            # record state
            result.append((z, q, current_n))

        return result

    def run_for_optimizer(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Math-Facing Forward Pass.
        Returns: A numpy array of shape (N_steps, 3) -> [z, w, R]
        """
        raw_history = self._run(params)
        wavelength = self.sources.wavelength
        
        # Convert to a flat array for efficient slicing in loss functions
        # autograd loves numpy arrays, hates lists of tuples.
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
        trace = [(z, GaussianBeam(q, wavelength)) for z, q, n in raw_history]

        return SimulationResult(trace)

