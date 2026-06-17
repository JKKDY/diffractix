import autograd.numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from autograd import jacobian
from typing import Tuple
from .beam import GaussianBeam


@dataclass
class SimulationResult:
    # List of (position, beam) tuples
    trace: List[Tuple[float, GaussianBeam]]

    @property
    def final_beam(self) -> GaussianBeam:
        """Helper to get the output beam quickly."""
        return self.trace[-1][1]

    def plot(self, points_per_segment: int = 50):
        """
        Generates a smooth matplotlib visualization of the beam propagation.
        """
        import matplotlib.pyplot as plt
        import numpy as np # Use standard numpy for plotting, not autograd
        
        if not self.trace:
            print("No trace data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Get wavelength from the first beam for the plot title and color
        wavelength = self.trace[0][1].wavelength
        color = 'red' if wavelength > 600e-9 else 'blue'
        
        # We look at pairs of points: (current, next)
        for i in range(len(self.trace) - 1):
            z_start, beam_start = self.trace[i]
            z_end, _ = self.trace[i+1]
            
            # If z doesn't change, it's a thin element (lens/interface) -> Skip
            if np.isclose(z_start, z_end):
                
                # Optional: draw a vertical line to indicate optical elements
                ax.axvline(z_start, color='k', linestyle='--', alpha=0.3)
                continue
                
            # Interpolate points across the propagation segment
            z_local = np.linspace(0, z_end - z_start, points_per_segment)
            
            # Calculate width at these local offsets 
            w_dense = np.array([beam_start.w_at_z(z) for z in z_local])
            z_global = z_start + z_local
            
            # Plot beam envelope
            ax.plot(z_global, w_dense * 1e3, color=color, alpha=0.8, lw=1) 
            ax.plot(z_global, -w_dense * 1e3, color=color, alpha=0.8, lw=1) 
            ax.fill_between(z_global, w_dense * 1e3, -w_dense * 1e3, color=color, alpha=0.1)
   
        # Formatting
        ax.set_xlabel("Position Z (m)")
        ax.set_ylabel("Beam Radius w(z) (mm)")
        ax.set_title(f"Gaussian Beam Propagation (\u03bb={wavelength*1e9:.0f} nm)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

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

    def __str__(self) -> str:
        """
        Produces a readable ASCII table of the beam propagation trace.
        Example:
        Z [m]      | w [mm]     | R [m]          
        -----------------------------------------
        0.0000     | 1.0000     | Inf            
        0.1000     | 1.0005     | 5.2000         
        """
        # Define column widths
        col_z = 10
        col_w = 10
        col_r = 15
        
        # Header
        header = f"{'Z [m]':<{col_z}} | {'w [mm]':<{col_w}} | {'R [m]':<{col_r}}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        
        for z, beam in self.trace:
            # Convert w to mm for readability
            w_mm = beam.w * 1000.0
            
            # Handle Infinity for R (Plane waves)
            R_val = beam.R
            if np.isinf(R_val) or abs(R_val) > 1e10:
                r_str = "Inf"
            else:
                r_str = f"{R_val:.4f}"
            
            # Format row
            lines.append(f"{z:<{col_z}.4f} | {w_mm:<{col_w}.4f} | {r_str:<{col_r}}")
            
        return "\n".join(lines)




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
                variable_params: list,
                constraints: list[callable] = None):
        assert len(sources) == 1, "Currently only single input beam supported."
                        
        # Physics
        self.steps = steps              
        self.initial_values = np.array(initial_values, dtype=float)
        self.sources = sources[0] 
        self.environment = environment

        # Metadata & Optimization Hooks
        self.parameter_transform = parameter_transform
        self.variable_params = variable_params
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

