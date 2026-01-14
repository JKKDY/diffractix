import inspect
import autograd.numpy as np
from collections.abc import Iterable
from dataclasses import dataclass

from .elements import OpticalElement, Space
from .beams import GaussianBeam
from .simulation import Simulation


@dataclass(frozen=True)
class ParameterInfo:
    """
    Metadata for a single flattened parameter in the simulation.
    Acts as a 'Phonebook' entry mapping indices to elements.
    """
    index: int           # Position in the flat np.ndarray
    element_id: str      # Unique ID of the parent element
    element_label: str   # Human-readable name (e.g., 'Lens 1')
    param_name: str      # Name of the variable (e.g., 'f', 'd', 'R')
    initial_value: float # Value at time of build
    is_variable: bool    # True if the user marked this for optimization


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


    def add(self, element: OpticalElement | Iterable[OpticalElement], z: float = None, optimize_z: bool = False):
        """
        Add an element to the optical path.
        Args:
            element: The OpticalElement (Lens, Mirror, etc.)
            z:      Optional Absolute Position. 
                    If None, places element relative to the previous one (immediately after).
                    If Set, inserts an 'AutoSpace' to reach this Z coordinate.
            optimize_z: If True, the thickness of the 'AutoSpace' created to reach 'z' becomes a Variable parameter.
        """

        # recurisve duck typing for sequences of elments
        if isinstance(element, Iterable):
            for el in element:
                self.add(el) 
            return self # allow chaining

        # single element case
        assert isinstance(element, OpticalElement), "Only OpticalElement instances can be added to the System."

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
        
        # validate absolute positions
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

        # validate refractive index transitions
        for i in range(len(self.elements) - 1):
            el1 = self.elements[i]
            el2 = self.elements[i+1]
            
            # Check for index mismatch between consecutive spaces
            if isinstance(el1, Space) and isinstance(el2, Space):
                if not np.isclose(el1.n, el2.n):
                    raise ValueError(
                        f"Refractive Index Mismatch at boundary between '{el1.label}' and '{el2.label}'.\n"
                        f"Medium 1: n={el1.n}, Medium 2: n={el2.n}.\n"
                        "To change index, you must explicitly insert a 'DielectricInterface'."
                    )


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
                    # shock absorber logic: This spacer must vary to absorb upstream changes
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
                    print(f"Generated Z-Constraint at Element '{el.label}' (z={absolute_pos}) to lock position after variable elements: {[e.label for e in variable_length_block]}")
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

        functions = []          # List of element step functions (returning ABCD matrices)
        length_functions = []   # 
        indices = []            # list of tuples of indices into parameter array. Each tuple corresponds to one element
        values = []             # (initial) parameter values
        param_defs = []         # parameter map: We need this so the Optimizer knows that Index 5 is "Lens 1 Focal Length"
    
        # track the number of variable parameters
        current_param_idx = 0
        
        # Loop over compiled (i.e. finalized) elements
        for el in  self._compiled_elements:
            if hasattr(el, 'get_sim_data'):
                func, len_func, vals = el.get_sim_data()
             
                # store Functions
                functions.append(func)
                length_functions.append(len_func)

                # store Indices & parameter values
                idxs = tuple(range(current_param_idx, current_param_idx + len(vals)))                    
                indices.append(idxs)
                values.extend(vals)
                
                # generate Metadata                
                for i, v in enumerate(vals):                    
                    param_defs.append(ParameterInfo(
                        index=current_param_idx + i,
                        element_id=el.id,
                        element_label=el.label,
                        param_name=el.param_names[i],
                        initial_value=v,
                        is_variable=(el.param_names[i] in el._variable_params)
                    ))

                current_param_idx += len(vals)

            else:
                raise ValueError(f"Element {el} missing get_sim_data.")
            
        # pass constraints and definitions to Simulation
        return Simulation(
            steps=functions, 
            length_steps=length_functions,
            param_indices=indices, 
            initial_params=values,
            input_beams=self.input_beams, 
            structure_metadata=self._generate_metadata(),
            param_definitions=param_defs,         # The Phonebook
            constraints=self._generated_constraints # The Z-Locks
        )


    def __str__(self):
        lines = []
        is_compiled = hasattr(self, '_compiled_elements') and self._compiled_elements
        
        if is_compiled:
            target_list = self._compiled_elements
            mode_label = "(Compiled / Resolved)"
        else:
            target_list = self.elements
            mode_label = "(Blueprint)"

        lines.append(f"=== Optical System Configuration {mode_label} ===")
        
        # 1. Inputs Section
        lines.append("\n--- Input Beams ---")
        if self.input_beams:
            for i, b in enumerate(self.input_beams):
                lines.append(f"  [{i}] {b}")
        else:
            lines.append("  (None)")

        # 2. Components Table
        lines.append("\n--- Component Schedule ---")
        
        # Added 'Z-Pos' column
        header = f"{'#':<4} {'Z-Pos':<10} {'Type':<15} {'Label':<20} {'Parameters':<50} {'Line'}"
        lines.append(header)
        lines.append("-" * len(header))

        current_z = 0.0
        for i, el in enumerate(target_list):
            # A. Basic Info
            el_type = el.__class__.__name__
            label = el.label if len(el.label) < 19 else el.label[:16] + "..."
            
            # B. Z-Positioning
            # Show current_z before adding the element's own length
            z_str = f"{current_z:.4g}m"
            
            # C. Source Metadata
            line_num = str(el._source_info.get('line', '-'))
            
            # D. Parameters & Values
            if hasattr(el, 'get_sim_data'):
                _, _, vals = el.get_sim_data()
                names = el.param_names
                
                param_chunks = []
                for j, p_val in enumerate(vals):
                    p_name = names[j] if j < len(names) else f"p{j}"
                    is_var = p_name in el._variable_params
                    val_str = f"{p_val:.4g}"
                    
                    status = ""
                    if is_var:
                        status = "[VAR]"
                    
                    # Highlight auto-generated logic
                    if "AutoSpace" in label and p_name == 'd':
                         status = "[AUTO-VAR]" if is_var else "[AUTO-FIX]"

                    param_chunks.append(f"{p_name}={val_str}{status}")
                
                params_str = ", ".join(param_chunks)
            else:
                params_str = "No Sim Data"

            # E. Append Row
            lines.append(f"{i:<4} {z_str:<10} {el_type:<15} {label:<20} {params_str:<50} {line_num}")
            
            # Increment Z for the next element in the table
            current_z += el.length

        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    x = Space(d=0.3, label="Slider").variable()
    print(x)