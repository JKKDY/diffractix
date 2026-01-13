import inspect
import autograd.numpy as np
from collections.abc import Iterable

from .elements import OpticalElement, Space
from .beams import GaussianBeam
from .simulation import Simulation

class System:
    """
    The declarative builder for the optical setup.
    It holds the 'Blueprints' before they are compiled into math.
    """
    def __init__(self):
        self.elements: list[OpticalElement] = []
        self.input_beams: list[GaussianBeam] = []
        
        # store "compiled" layout (after self.build())
        self._compiled_elements : list[OpticalElement] = []
        self._generated_constraints : list[callable] = []

        # Track execution counts for UI sync (Loop disambiguation)
        self._execution_counts : dict = {}

    # TODO rename to add_input_beam
    def input(self, beam_or_beams: GaussianBeam | list[GaussianBeam]):
        """Define the source beam(s) entering the system at z=0."""
        if isinstance(beam_or_beams, list):
            raise NotImplementedError("Multiple input beams not yet supported.")
            self.input_beams.extend(beam_or_beams)
        else:
            self.input_beams.append(beam_or_beams)
        return self 


    def add(self, element: OpticalElement, z: float = None, optimize_z: bool = False):
        """
        Add an element to the optical path.
        Args:
            element: The OpticalElement (Lens, Mirror, etc.)
            z:      Optional Absolute Position. 
                    If None, places element relative to the previous one (immediately after).
                    If Set, inserts an 'AutoSpace' to reach this Z coordinate.
            optimize_z: If True, the 'AutoSpace' created to reach 'z' becomes a Variable parameter.
        """

        # reflection magic (For UI Sync later)
        # we capture where in the user's script this component was added.
        frame = inspect.currentframe().f_back
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        
        # Track loop index: # times the line was executed
        call_index = self._execution_counts.get(lineno, 0)
        self._execution_counts[lineno] = call_index + 1
        
        # attach metadata to the element   
        element._source_info.update({
            "file": filename,
            "line": lineno,
            "call_index": call_index
        })

        # now add element
        if z is not None: 
            element.z = z 
            element.optimize_z = optimize_z
        self.elements.append(element)
        return self


    def _validate_layout(self):
        """
        Validates user input:
            1. Ensures that absolute positions increase monotonically
            2. Ensures that absolute positions do not conflict with the accumulated length of relative elements
        """
        # tracks our position in the simulation
        current_z = 0.0
        
        for i, el in enumerate(self.elements):

            # if element has absolute position, make sure it is not behind the current cursor
            absolute_pos = getattr(el, 'z', None)
            if absolute_pos is not None:
                # tolerance of 1nm to avoid floating point issues
                if absolute_pos < current_z - 1e-9:
                    raise ValueError(
                        f"Layout Error at Element #{i} ({el.label}): "
                        f"Requested absolute z={absolute_pos:.4f}, but previous elements "
                        f"push the optical path to z={current_z:.4f}."
                    )
                # set curosr to new absolute position
                current_z = absolute_pos

            # add length of this element to the cursor 
            # (ensures that relative elements fit between absolute elements)
            current_z += el.length


    def _resolve_layout(self):
        """
        Compiles the mixed Absolute/Relative list into a pure Sequential list.
          - Inserts AutoSpaces to reach Absolute Z positions.
          - Configures optimization for those spaces (Variable vs Fixed).
          - Generates constraints if Fixed Anchors follow Variable Elements.
        """
        # clear any previous state
        self._compiled_elements = []
        self._generated_constraints = [] 
        
        current_z = 0.0
        variable_length_block = [] # all variable length elements since last fixed absolute
        
        for el in self.elements:
            absolute_pos = getattr(el, 'z', None)
            opt_absolute_pos = getattr(el, 'optimize_z', False)
            
            # if element has absolute position, we may need to insert a spacer
            if absolute_pos is not None:
                gap = absolute_pos - current_z

                # spacer inserted right before the absolute element to fill gap
                spacer = Space(d=gap, label=f"AutoSpace_to_{absolute_pos}")

                # Check if chain is currently loose
                upstream_is_variable = len(variable_length_block) > 0

                if opt_absolute_pos:
                    # user explicitly requested optimization: spacer is variable & everything updstream is now loose 
                    # i.e. will be moved if spacer length is adjusted
                    spacer.variable('d')
                    last_variable_length_elem = spacer
                    variable_length_block.append(spacer)

                elif not opt_absolute_pos and len(variable_length_block) > 0: # 
                    # element is Fixed Absolute, but upstream is Variable -> we need to lock this position
                    # shock absorber logic: This spacer MUST vary to absorb upstream changes
                    spacer.variable('d')

                    # we do this by adding a constraint: 
                    target_trace_idx = len(self._compiled_elements) + 1
                    def make_z_constraint(idx, target_z, lbl):
                        # Capture specific values in closure
                        def z_lock(params, simulation_trace: list):
                            # data shape is (Steps+1, 3) -> [z, w, R]
                            actual_z = simulation_trace[idx, 0]
                            return (actual_z - target_z) 
                        return z_lock

                    self._generated_constraints.append(
                        make_z_constraint(target_trace_idx, absolute_pos, el.label)
                    )
                    
                    # break the varible length block
                    variable_length_block = []
                    

                self._compiled_elements.append(spacer)
                current_z += gap

            # add actual element
            self._compiled_elements.append(el)
            current_z += el.length

           
            if el.has_variable_length:
                variable_length_block.append(el)
        
        return self._compiled_elements



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
        # AI generated. Hope it works.
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
