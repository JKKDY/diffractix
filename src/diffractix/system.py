import inspect
import autograd.numpy as np
from collections.abc import Iterable
from dataclasses import dataclass, fields

from .elements import OpticalElement, Space, Interface, ABCD
from .beams import GaussianBeam
from .simulation import Simulation, SimulationStep
from .graph import Node, Parameter, Symbol, InputNode, compile_parameter_transform, Constant



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


@dataclass(frozen=True)
class Environment:
    ambient_n: Parameter

    @property
    def variables(self):
        return tuple(f.name for f in fields(self))


class System:
    """
    The declarative builder for the optical setup.
    It holds the 'Blueprints' before they are compiled into math.
    """
    def __init__(self, ambient_n: float = 1.0, ambient_n_variable: bool = False):
        self.environment = Environment(
            ambient_n = Parameter(value=ambient_n, name="ambient_n", fixed=not ambient_n_variable)
        )

        self.elements: list[OpticalElement] = []
        self.input_beams: list[GaussianBeam] = []
        
        # store "compiled" layout (after self.build())
        self._compiled_elements : list[OpticalElement] = []
        self._generated_constraints : list[callable] = []

        # Track execution counts for UI sync (Loop disambiguation)
        self._execution_counts : dict = {}

    def add_input(self, beam_or_beams: GaussianBeam | list[GaussianBeam]):
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


    def _bind_environment_variables(self, elements):
        for el in elements:
            for param in el.param_names:
                node = getattr(el, param)
                if isinstance(node, InputNode):
                    node = node.node

                if isinstance(node, Symbol) and node.name in self.environment.variables:
                    node.bind(getattr(self.environment, node.name))
        

    def _validate_layout(self):
        """
        Validates user input:
            1. Ensures that absolute positions increase monotonically
            2. Ensures that absolute positions do not conflict with the accumulated length of relative elements
        """
        # 1. step: ensure every parameter is an AST node or None. ensure they are all convertable to floats
        for i, el in enumerate(self.elements):
            for name in el.param_names:
                param = getattr(el, name)

                if not isinstance(param, Node) and param is not None:
                    raise ValueError(f"Encountered a non Node parameter {name} of element {el}: {param}. Type: {type(param)}")

        # 2. step: validate absolute positions:
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

        # validate refractive index transitions
        current_index = self.environment.ambient_n.value

        for i, el in enumerate(self.elements):
            if isinstance(el, Space):
                space_n = float(el.n.value)
                
                if not np.isclose(current_index, space_n):
                     raise ValueError(
                        f"Refractive Index Mismatch at Element #{i} ({el.label}).\n"
                        f"Current Field Index: n={current_index:.4f}\n"
                        f"Space Medium Index:  n={space_n:.4f}\n"
                        "Physics Violation: Discontinuous index change detected.\n"
                        "Solution: Insert an Interface(n1=..., n2=...) before this Space."
                    )
            elif isinstance(el, Interface):
                current_index = el.n2.value

            elif isinstance(el, ABCD):
                try:
                    val = float(el.n.value) if hasattr(el.n, 'value') else float(el.n)
                    current_index = val
                except (TypeError, ValueError):
                    pass # Inherited index, no update


    def _resolve_layout(self):
        """
        Compiles the mixed Absolute/Relative list into a pure Sequential list.
          - Inserts AutoSpaces to reach Absolute Z positions.
          - Configures optimization for those spaces (Variable vs Fixed).
          - shock absorber contraints (for absolute positioning) are baked into the graph topology during this step.
        """
        # clear any previous state
        self._compiled_elements = []

        current_z_node = Constant(0.0)
        
        for el in self.elements:
            absolute_pos = getattr(el, 'z', None)
            opt_absolute_pos = getattr(el, 'optimize_z', False)
            
            # if element has absolute position, we need to insert a spacer
            # the width of the spacer is dependent on the position of the prevous element (given by current_z_node) 
            # and the desired position of the new element. If the user selects to have the abs psotion variable
            # the spacers length must be variable as well (i.e. Parameter with fixed = false)
            if absolute_pos is not None:
                
                if opt_absolute_pos:
                    # position of the element is variable
                    target_z_node = Parameter(absolute_pos, name=f"Anchor_{el.label}", fixed=False)
                else:
                    # Fixed Anchor.
                    target_z_node = Constant(absolute_pos)
                
                # construct a hard constraint for the width of the spacer
                # this links the spacer directly to all previous upstream nodes.
                gap_node = target_z_node - current_z_node
                spacer = Space(d=gap_node, label=f"AutoSpace_to_{absolute_pos}")
                
                self._compiled_elements.append(spacer)
                
                # Update Cursor
                # We add the spacer to our running totals. 
                # Mathematically: New_Sum = Target_Z
                current_z_node = current_z_node + spacer.element_length 

            # add actual element and update cursor
            self._compiled_elements.append(el)
            current_z_node = current_z_node + el.element_length


    def _generate_metadata(self):
        pass
    
    def _build_simulation(self):
        def normalize_idx_func(idx_func):
            # normalize the index functions. 
            # The normalized index function depends on the refractive index of the previous element (curr)
            if idx_func is not None: return lambda curr, *args, idx_func=idx_func: idx_func(*args)
            else: return lambda curr, *args: curr

        sim_steps = []
        roots = []
        
        # compile elements into steps
        # We process elements linearly to build the "Physics Schedule"
        current_param_idx = 0
        for el in self._compiled_elements:
            mat_func, len_func, idx_func = el.get_sim_functions()
            
            # Identify which slice of the full input list belongs to this element
            num_params = len(el.parameters) # el.parameters is the list of InputNodes
            idxs = tuple(range(current_param_idx, current_param_idx + num_params))
            
            step = SimulationStep(
                matrix_func=mat_func,
                length_func=len_func,
                index_func=normalize_idx_func(idx_func),
                param_indices=idxs
            )
            sim_steps.append(step)
            
            # TODO temporary fix to ABCD.n = None problem
            for name in el.param_names:
                param = getattr(el, name)
                if param is None:
                    setattr(el, name, Constant(1))
                    
            roots.extend(el.parameters)
            current_param_idx += num_params


        # compile graph logic
        transform_func, initial_values, variable_params = generate_parameter_transform(roots)


        # pass constraints and definitions to Simulation
        return Simulation(
            steps=sim_steps, 
            sources=self.input_beams, 
            environment = self.environment,
            initial_values=initial_values,
            parameter_transform = transform_func,
            constraints=self._generated_constraints # The Z-Locks
        )


    def build(self) -> 'Simulation':
        """
        Compiles the element list into a flat, differentiable Simulation.
        """
        self._bind_environment_variables(self.elements)
        self._validate_layout()
        self._resolve_layout() 
        return self._build_simulation()
      


    def __str__(self):
        # 1. Determine Source
        is_compiled = len(self._compiled_elements) > 0
        elements_list = self._compiled_elements if is_compiled else self.elements

        # 2. Header Formatting
        status = "COMPILED/RESOLVED" if is_compiled else "BLUEPRINT (UNRESOLVED)"
        title = f" Optical System State [{status}] "
        
        # Added 'Z [mm]' column
        header = f"{'ID':<4} | {'Z [m]':<12} | {'Type':<15} | {'Label':<20} | {'L [m]':<10} | {'n':<8} | {'Parameters'}"
        divider = "-" * 125
        
        lines = [
            "",
            f"{title:=^125}",
            header,
            divider
        ]

        # 3. Iterate and Format
        current_z = 0.0 # Tracking cursor for compiled systems
        
        for i, el in enumerate(elements_list):
            # --- Type & Label ---
            el_type = el.__class__.__name__
            el_label = (el.label[:17] + '..') if el.label and len(el.label) > 19 else (el.label or "")

            # --- Length ---
            try:
                l_val = el.length
                l_str = f"{l_val:.4g}"
            except Exception:
                l_val = 0.0 # Fallback for Z calculation if length is broken
                l_str = "Err"

            # --- Z Position Logic ---
            if is_compiled:
                # In compiled mode, we strictly trace the path
                z_str = f"{current_z:.4f}"
                current_z += l_val
            else:
                # In blueprint mode, we check for explicit intent
                abs_z = getattr(el, 'z', None)
                if abs_z is not None:
                    z_str = f"@{abs_z:<.4g}" # Mark as absolute
                else:
                    z_str = "Rel"      # Mark as relative

            # --- Refractive Index ---
            try:
                n_val = el.refractive_index
                n_str = f"{n_val:.4f}" if n_val is not None else "-"
            except Exception:
                n_str = "?"

            # --- Parameters ---
            param_parts = []
            try:
                _, _, _, val_nodes = el.get_sim_data()
                current_vals = [float(x) for x in val_nodes]
                vars_set = set(el.variable_parameter_names)

                for name, val in zip(el.param_names, current_vals):
                    tag = "[VAR]" if name in vars_set else "" 
                    param_parts.append(f"{name}={val:<.4g}{tag}")
            except Exception:
                param_parts.append("Uninitialized")

            params_str = " ".join(param_parts)

            # --- Row Assembly ---
            lines.append(f"{i:<4} | {z_str:<12} | {el_type:<15} | {el_label:<20} | {l_str:<10} | {n_str:<8} | {params_str}")

        lines.append(divider)
        if is_compiled:
            lines.append(f"Total System Length: {current_z:.4f} m")
        lines.append("")
        
        return "\n".join(lines)
