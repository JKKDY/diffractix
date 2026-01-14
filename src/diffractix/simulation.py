import autograd.numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from autograd import jacobian
from typing import Tuple
from . import core
from .beams import GaussianBeam


@dataclass
class SimulationResult:
    # List of (position, beam) tuples
    trace: List[Tuple[float, GaussianBeam]]
    structure_metadata: List[Dict]

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
            Rs.append(float(beam.curvature_at_z(0)))

        return {
            "structure": self.structure_metadata,
            "results": [{
                "wavelength": wavelength,
                "z": zs,
                "w": ws,
                "R": Rs
            }]
        }



class Simulation:
    """
    The compiled, executable physics engine.
    This class is 'stateless' regarding the physics variables. 
    It defines the TOPOLOGY of the system.
    """
    def __init__(self, 
                 steps: list[callable], 
                 length_steps: list[callable],
                 param_indices: list[tuple], 
                 initial_params: np.ndarray, 
                 input_beams: list['GaussianBeam'],
                 structure_metadata: list[dict],
                 param_definitions: list['ParameterDef'],
                 constraints: list[callable] = None):
        # validation
        assert len(steps) == len(param_indices), "Steps and param_indices must align."
        assert len(steps) == len(length_steps), "Steps and length_steps must align."
        assert len(input_beams) == 1, "Currently only single input beam supported."
        
        # Physics
        self.steps = steps               # functions returning ABCD matrices
        self.length_steps = length_steps # functions returning physical length
        self.param_indices = param_indices # List of tuples mapping steps to indices in params array
        self.initial_params = np.array(initial_params, dtype=float)
        self.input_beam = input_beams[0] 

        # Metadata & Optimization Hooks
        self.structure_metadata = structure_metadata
        self.param_definitions = param_definitions
        self.constraints = constraints if constraints is not None else []


    def _run(self, params:np.ndarray ) -> List[Tuple[float, complex]]:
        if params is None:
            params = self.initial_params

        all_beams_history = []

        # initialize State
        q = self.input_beam.q
        z = 0.0  # starting at z=0
        wavelength = self.input_beam.wavelength
        
        # Initialize history with starting beam
        result = [(z, q)]

        # Propagation Loop
        for func, len_func, idx in zip(self.steps, self.length_steps, self.param_indices):

            # extract parameters
            step_params = params[list(idx)]
            mat = func(*step_params)
            step_z = len_func(*step_params)
        
            # propagate with möbius transform
            A, B = mat[0]
            C, D = mat[1]
            numerator = A * q + B
            denominator = C * q + D
                     
            q = numerator / denominator
            z += step_z

            # record state
            result.append((z, q))

        return result

    def run_for_optimzer(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Math-Facing Forward Pass.
        Returns: A numpy array of shape (N_steps, 3) -> [z, w, R]
        """
        raw_history = self._run(params)
        wavelength = self.input_beam.wavelength
        
        # Convert to a flat array for efficient slicing in loss functions
        # autograd loves numpy arrays, hates lists of tuples.
        data = []
        for z, q in raw_history:
            w, R = core.w_R_from_q(q, wavelength)
            data.append([z, w, R])
            
        return np.stack(data) # Shape: (Steps, 3)

    def foo(self):
        print("bar")


    def run(self, params: np.ndarray | None = None) -> 'SimulationResult':
        """
        User-Facing Forward Pass.
        Returns: Rich objects for the UI/Scripting.
        """
        raw_history = self._run(params)
        wavelength = self.input_beam.wavelength
        
        # Convert raw (z, q) -> (z, GaussianBeam) for the Result object
        trace = [(z, GaussianBeam(q, wavelength)) for z, q in raw_history]

        return SimulationResult(trace, self.structure_metadata)

