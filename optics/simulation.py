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
                 steps: List[callable], 
                 length_steps: List[callable],
                 param_indices: List[int], 
                 initial_params: np.ndarray, 
                 input_beams: List['GaussianBeam'],
                 structure_metadata: List[Dict]):

        assert len(steps) == len(param_indices), "Steps and param_indices must align."
        assert all(isinstance(b, GaussianBeam) for b in input_beams), "Input beams must be GaussianBeam instances."
        assert len(input_beams) == 1, "Currently only single input beam supported."
        
        self.steps = steps                              # List of step functions (returning ABCD matrices)       
        self.length_steps = length_steps
        self.param_indices = param_indices              # Map: Step i -> Param j (or set of j's if matrix has multiple params)
        self.initial_params = np.array(initial_params, dtype=float)

        self.input_beam = input_beams[0] 
        self.structure_metadata = structure_metadata # for export logic later on


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
            if isinstance(idx, tuple):  
                # Multi-parameter element (e.g. generic ABCD)
                step_params = params[list(idx)]
                mat = func(*step_params)
                step_z = len_func(*step_params)
            else:
                # Single-parameter element (e.g. Lens, Space)
                val = params[idx]
                mat = func(val)
                step_z = len_func(val)

            # propagate
            q = core.propagate(q, mat)
            z += step_z

            # record state
            result.append((z, q))

        return result

    def run_for_optimzer(self, params: Optional[np.ndarray] = None) -> np.ndarray:
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


    def run(self, params: Optional[np.ndarray] = None) -> 'SimulationResult':
        """
        User-Facing Forward Pass.
        Returns: Rich objects for the UI/Scripting.
        """
        raw_history = self._run(params)
        wavelength = self.input_beam.wavelength
        
        # Convert raw (z, q) -> (z, GaussianBeam) for the Result object
        trace = [(z, GaussianBeam(q, wavelength)) for z, q in raw_history]

        return SimulationResult(trace, self.structure_metadata)


    def compute_jacobian(self, params: Optional[np.ndarray] = None, mode='w'):
        """
        Calculates the Jacobian matrix of the beam properties w.r.t parameters.
        
        Args:
            params: The parameters to evaluate at.
            mode: 'w' for Spot Size Jacobian, 'R' for Curvature, 'all' for both.
            
        Returns:
            Matrix of shape (Num_Steps, Num_Params)
        """
        if params is None: params = self.initial_params
        
        # pure function that isolates the output we care about
        def target_func(p):
            # Shape (Steps, 3) -> Columns: z, w, R
            results = self.run_for_optimizer(p) 
            
            if mode == 'w':
                return results[:, 1] # Return all 'w' values
            elif mode == 'R':
                return results[:, 2] # Return all 'R' values
            else:
                return results[:, 1:] # Return w and R flattened

        jac_func = jacobian(target_func)
        return jac_func(params)