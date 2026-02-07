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
            for node in el.parameters + [el.element_length, el.element_refractive_index]:
                if isinstance(node, InputNode):
                    node = node.node

                if isinstance(node, Symbol) and node.name in self.environment.variables:
                    node.bind(getattr(self.environment, node.name))
        

    def _validate_layout(self, elements):
        """
        Validates user input:
            1. Ensures that absolute positions increase monotonically
            2. Ensures that absolute positions do not conflict with the accumulated length of relative elements
        """
        # 1. step: ensure every parameter is an AST node or None. ensure they are all convertable to floats
        for i, el in enumerate(elements):
            for name in el.param_names:
                param = getattr(el, name)

                if not isinstance(param, Node) and param is not None:
                    raise Logic(f"Encountered a non Node parameter {name} of element {el}: {param}. Type: {type(param)}")

        # 2. step: validate absolute positions:
        current_z = 0.0
        for i, el in enumerate(elements):
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



    def _resolve_layout(self, elements):
        """
        Compiles the mixed Absolute/Relative list into a pure Sequential list.
          - Inserts AutoSpaces to reach Absolute Z positions.
          - Configures optimization for those spaces (Variable vs Fixed).
          - shock absorber contraints (for absolute positioning) are baked into the graph topology during this step.
        """
        # clear any previous state
        resolved_elements = []

        current_z_node = Constant(0.0)
        
        for el in elements:
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
                
                resolved_elements.append(spacer)
                
                # Update Cursor: We add the spacer to our running totals. 
                current_z_node = current_z_node + spacer.element_length 

            # add actual element and update cursor
            resolved_elements.append(el)
            current_z_node = current_z_node + el.element_length

        return resolved_elements



    def _resolve_refractive_indices(self, elements, auto_insert_interface: bool = False):
        """
        Wires the refractive index graph.
        1. Binds 'inherit' sockets (InputNode(Symbol(None))) to the upstream index.
        2. Validates continuity for fixed-index elements (Spaces).
        3. Auto-inserts Interfaces if requested/needed.
        """
        resolved_elements = []
        
        # State tracking: The AST Node representing the index at the current z-plane
        # Start with the environment ambient index
        current_index_node = self.environment.ambient_n
        for i, el in enumerate(elements):
            index_handle = el.element_refractive_index
            
            # InputNode(None) -> inherit from previous element
            if isinstance(index_handle, InputNode) and index_handle.node is None:
                el.element_refractive_index.node = current_index_node

            # ABCD and Interfaces are allowed to change the index 
            # (in the case of ABCD we trust the user knows what they are doing)
            elif isinstance(el, (Interface, ABCD)):
                current_index_node = index_handle

            # element does not have `inherit` socekt + is not ABCD or Interface -> check for continuity
            else:
                target_val = index_handle.value
                current_val = current_index_node.value
                
                # Check for physical violation
                if not np.isclose(current_val, target_val):
                    
                    if auto_insert_interface:
                        # Auto-Fix: Create an Interface to bridge the gap
                        interface = Interface(n1=current_index_node, n2=index_handle)
                        resolved_elements.append(interface)                        
                    else:
                        raise ValueError(
                            f"Refractive Index Mismatch at Element #{i} ({el.label}).\n"
                            f"System Flow: n={current_val:.4f}\n"
                            f"Element Demand: n={target_val:.4f}\n"
                            "Physics Violation: Discontinuous index change detected.\n"
                            "Solution: Insert an Interface or enable auto_insert_interface."
                        )
                current_index_node = index_handle

            resolved_elements.append(el)
        
        return resolved_elements


    def _generate_metadata(self):
        pass


    def _build_simulation(self):
        roots = []
        sim_steps = []
        
        # "compile" elements into simulation steps
        for el in self._compiled_elements:
            start_idx = len(roots)
            p_nodes = el.parameters
            roots.extend(p_nodes)

            if el.element_length in el.parameters:
                length_idx = start_idx + p_nodes.index(el.element_length)
            else:
                length_idx = len(roots)
                roots.append(el.element_length)

                
            if el.element_refractive_index in el.parameters:
                index_idx = start_idx + p_nodes.index(el.element_refractive_index)
            else:
                index_idx = len(roots)
                roots.append(el.element_refractive_index)
            
            step = SimulationStep(
                compute_matrix=el.compute_matrix,
                param_indices=slice(start_idx, start_idx + len(p_nodes)),
                length_index=length_idx,
                index_index=index_idx
            )
            sim_steps.append(step)

        # compile parameter graph
        transform_func, initial_theta, variable_params = compile_parameter_transform(roots)
        
        return Simulation(
            steps=sim_steps,
            sources=self.input_beams,
            initial_values=initial_theta,
            parameter_transform=transform_func,
            constraints=self._generated_constraints
        )


    def build(self) -> 'Simulation':
        """
        Compiles the element list into a flat, differentiable Simulation.
        """
        self._bind_environment_variables(self.elements)
        self._validate_layout(self.elements)
        resolved = self._resolve_layout(self.elements) 
        resolved = self._resolve_refractive_indices(resolved) 

        self._compiled_elements = resolved

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
