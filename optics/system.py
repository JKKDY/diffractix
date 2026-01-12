import inspect
import autograd.numpy as np
from typing import Union, List, Optional
from collections.abc import Iterable

from .elements import OpticalElement, Lens, Space
from .beams import GaussianBeam
from .simulation import Simulation

class System:
    """
    The declarative builder for the optical setup.
    It holds the 'Blueprints' before they are compiled into math.
    """
    def __init__(self):
        # We don't define wavelength here. It lives on the Input Beam.
        self.elements: List[OpticalElement] = []
        self.input_beams: List[GaussianBeam] = []
        
        # Track execution counts for UI sync (Loop disambiguation)
        self._execution_counts = {}


    def input(self, beam_or_beams: Union[GaussianBeam, List[GaussianBeam]]):
        """Define the source beam(s) entering the system at z=0."""
        if isinstance(beam_or_beams, list):
            raise NotImplementedError("Multiple input beams not yet supported.")
            self.input_beams.extend(beam_or_beams)
        else:
            self.input_beams.append(beam_or_beams)
        return self 

    def add(self, element: Union[OpticalElement, List[OpticalElement]], z: float = None):
        """
        Add component(s) to the optical path.
        Captures source code metadata for the UI Patcher.
        """
        if z is not None: 
            raise NotImplementedError("Absolute Z placement is not yet implemented.")

        # handle lists (recursive add)
        if isinstance(element, list):
            if z is not None:
                raise ValueError("Cannot apply a single absolute Z to a list of elements.")
            for e in element:
                self.add(e)
            return self

        # Reflection Magic (For UI Sync)
        # We capture where in the user's script this component was added.
        frame = inspect.currentframe().f_back
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        
        # Track loop index: # times the line was executed
        call_index = self._execution_counts.get(lineno, 0)
        self._execution_counts[lineno] = call_index + 1
        
        # Attach metadata to the element   
        element._source_info.update({
            "file": filename,
            "line": lineno,
            "call_index": call_index
        })

        # if user specified absolute z, set it
        if z is not None: 
            element.z = z 
        self.elements.append(element)
        return self


    def _validate_layout(self):
        """
        Validates that the element layout is consistent.
        E.g., of only relative elements are used, there must be a space between each
        """
        # not implement yet
        # TODO validate inputs
        pass


    def _resolve_layout(self):
        """
        Converts a mix of Absolute and Relative elements into a 
        purely Relative (Sequential) list for the simulation.
        """
        # not implement yet
        # TODO enable absolute positioning
        pass

    def _generate_metadata(self):
        pass


    def build(self) -> 'Simulation':
        """
        Compiles the element list into a flat, differentiable Simulation.
        """
        self._validate_layout()
        self._resolve_layout()

        functions = [] # List of element step functions (returning ABCD matrices)
        indices = []  # Can contain int tuple of ints (in case of multi-parameter elements)
        values = []  # parameter values
        length_functions = []

        # we start indexing parameters at 0
        current_param_idx = 0
        
        for el in self.elements:
            if hasattr(el, 'get_sim_data'):
                func, len_func, val = el.get_sim_data()
                functions.append(func)
                length_functions.append(len_func)

                # check for multi parameter element
                if isinstance(val, (list, tuple, np.ndarray)):
                    # extend with chunk of indices
                    idxs = tuple(range(current_param_idx, current_param_idx + len(val)))                    
                    indices.append(idxs)
                    values.extend(val) # flatten values into the big array
                    current_param_idx += len(val)
                    
                # single parameter element
                else:
                    indices.append(current_param_idx) 
                    values.append(val)
                    current_param_idx += 1
            else:
                raise ValueError(f"Element {el} does not implement get_sim_data().")
           
        # create immutable simulation object
        return Simulation(
            steps=functions, 
            length_steps=length_functions,
            param_indices=indices, 
            initial_params=values,
            input_beams=self.input_beams, 
            structure_metadata=self._generate_metadata()
        )


    def __str__(self):
        lines = []
        lines.append("=== Optical System Configuration ===")
        
        # 1. Inputs Section
        lines.append("\n--- Input Beams ---")
        if self.input_beams:
            for i, b in enumerate(self.input_beams):
                lines.append(f"  [{i}] {b}")
        else:
            lines.append("  (None)")

        # 2. Components Table
        lines.append("\n--- Component Schedule ---")
        
        # Define Columns: Index | Type | Label | Parameters (Value + Status) | Source Line
        # We allow a wide column for parameters to handle matrices
        header = f"{'#':<4} {'Type':<12} {'Label':<15} {'Parameters':<55} {'Line'}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, el in enumerate(self.elements):
            # A. Basic Info
            el_type = el.__class__.__name__
            # Truncate label if too long for cleaner table
            label = el.label if len(el.label) < 15 else el.label[:12] + "..."
            
            # B. Source Metadata (Captured in System.add)
            # Default to '?' if not found
            line_num = str(el._source_info.get('line', '?'))
            
            # C. Parameters & Values
            # We assume the element has the 'get_sim_data' and 'param_names' we defined earlier
            if hasattr(el, 'get_sim_data'):
                _, _, val = el.get_sim_data()
                
                # Handle scalar vs list/array (e.g. ABCD matrix)
                if isinstance(val, (list, tuple, np.ndarray)):
                    vals = np.array(val).flatten()
                else:
                    vals = [val]
                
                # Get names safely
                names = getattr(el, 'param_names', [f"p{j}" for j in range(len(vals))])
                
                # Build parameter string: "f=0.10 [VAR], d=0.50 [FIX]"
                param_chunks = []
                for p_name, p_val in zip(names, vals):
                    # Check if Variable
                    is_var = False
                    if hasattr(el, 'variable_params'):
                        is_var = p_name in el.variable_params
                    
                    # Format Value (Compact float)
                    val_str = f"{p_val:.4g}"
                    
                    # Status Tag
                    # We can make [VAR] visually distinct if supported, but text is safer
                    status = "[VAR]" if is_var else "[FIX]"
                    
                    param_chunks.append(f"{p_name}={val_str} {status}")
                
                params_str = ", ".join(param_chunks)
            else:
                params_str = "No Sim Data"

            # D. Append Row
            lines.append(f"{i:<4} {el_type:<12} {label:<15} {params_str:<55} {line_num}")

        return "\n".join(lines) + "\n\n"
