"""
Defines the base abstractions for all optical components in the simulation.

This module provides the `OpticalElement` base class, which handles the 
interface between physical components (Lenses, Spaces, etc.) and the 
differentiable simulation engine. It manages parameter variability, 
physical length tracking, and metadata generation.
"""

import inspect

from typing import Iterable, ClassVar
from dataclasses import dataclass, field
from abc import abstractmethod
from autograd import grad
import autograd.numpy as np
import itertools

from ..graph.ast import Node, Parameter, Constant, Symbol, BinaryOp, UnaryOp, InputNode,ASTNode


@dataclass
class OpticalElement:
    """
    Abstract base class for all optical components.

    An OpticalElement represents a physical entity in an optical system that 
    can transform a beam (via an ABCD matrix) and occupies physical space. 
    
    Key Features:
    - Differentiability: Provides matrix and length functions compatible 
      with autograd for optimization.
    - Variability: Parameters can be marked as 'variable' to be included 
      in optimization search spaces.
    - Layout Intelligence: Automatically determines if its physical length 
      is sensitive to parameter changes to assist in layout resolution.
    """

    # housekeeping
    id: int = field(init=False)
    label: str = None 
    _counts: ClassVar[dict] = {} # ClassVar so dataclasses doesn't treat this as an instance field
    _source_info: dict = field(default_factory=dict, repr=False)

    _id_counter: ClassVar[itertools.count] = itertools.count()

    #-----------------------
    # DUNDER / MAGIC METHODS
    #-----------------------
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Perform signature inspection at class construction time
        func = getattr(cls, "get_matrix", None)
        if func is None:
            raise TypeError("Subclass must define get_matrix")

        sig = inspect.signature(func)

        cls._param_names = [
            name for name in sig.parameters if name != "self"
        ]

       
    def __post_init__(self):
        self.id = next(self._id_counter)

        # set label to *class_name*#instantaitions 
        if self.label is None:
            cls = self.__class__
            
            if cls not in OpticalElement._counts:
                OpticalElement._counts[cls] = 0
            
            OpticalElement._counts[cls] += 1
            self.label = f"{cls.__name__}{OpticalElement._counts[cls]}"


    def __setattr__(self, name, value):
        def convertable_to_float(x):
            try: float(x)
            except (TypeError, ValueError): return False
            return True

        # only intercept access to parameters 
        if name not in getattr(self.__class__, '_param_names', set()):
            super().__setattr__(name, value)
            return


        # get existing handle
        current_handle = self.__dict__.get(name)

        # 1. Handle Update (Handle already exists)
        if isinstance(current_handle, InputNode):
            if isinstance(value, Node):
                # swap out the 'body'
                current_handle.node = value
            elif convertable_to_float(value):
                # Swap the 'Body' to a new Parameter. 
                current_handle.node = Parameter(value=value, name=name, fixed=True, owner=self)
            else:
                raise TypeError(f"Cannot assign {type(value)} to {name}")
            return

        # 2. Handle Initialization (First time)
        # This occurs during __init__ or dataclass default assignment
        if isinstance(value, Node):
            # Already a node (e.g. Lens(f=Symbol("x")))
            new_handle = InputNode(value)
        elif convertable_to_float(value):
            # Raw number (e.g. Lens(f=0.1))
            new_node = Parameter(value=value, name=name, fixed=True, owner=self)
            new_handle = InputNode(new_node)
        elif value is None:
            new_handle = None
        else:
            raise TypeError(f"Initial value for {self}.{name} must be Node or float, got {type(value)}")
        
        super().__setattr__(name, new_handle)



    def __str__(self):
        # Identity and length
        name = f"{self.__class__.__name__}"
        label_str = f" '{self.label}'" if self.label else ""
        header = f"{name:<15}{label_str:<20}L={self.length:.4g} {'[VAR]' if self.has_variable_length else '[FIX]'}"
        
        # parameters
        current_vals = self.values
        
        param_details = []
        for name, val in zip(self.param_names, current_vals):
            # Handle formatting for None vs Float
            val_str = f"{val:.4g}" if isinstance(val, (float, int)) else str(val)
            
            status = "[VAR]" if name in self.variable_parameter_names else "[FIX]"
            param_details.append(f"{name}={val_str} {status}")
        
        params_str = " | " + " | ".join(param_details)
      
        return f"{header:<50} {params_str}"
  

    #-----------
    # PROPERTIES 
    #-----------
    @property
    def variable_parameter_names(self):
        """
        Track which parameters are optimizable (e.g., {'f'}, {'d'}, {'A', 'D'})
        """
        names = []
        for name in self.param_names: 
            obj = getattr(self, name)
            if obj is not None and obj.is_constant is False:
                names.append(name)
        return names

    @property
    def param_names(self) -> list[str]:
        """
        Returns the list of parameter names corresponding to the simulation data.
        The order mathes that of self.get_sim_functions()'s returned parameters (4. return value).

        e.g. Lens.get_matrix(self, f) -> returns ['f']
        e.g. ABCD.get_matrix(self, A, B, C, D) -> returns ['A', 'B', 'C', 'D']
        """
        return self.__class__._param_names

    @property
    def parameters(self) -> tuple[ASTNode]:
        return (getattr(self, name) for name in self.param_names)

    @property
    def has_variable_length(self) -> bool:
        """
        Returns True if the physical length of this element changes when any of its currently variable parameters change.

        Strategies:
          1. Fast Fail: No parameters are variable.
          2. Explicit Fast-Path: Checks length_param_names (if provided).
          3. Implicit Fallback: autograd sensitivity check.
        """
        variable_params = self.variable_parameter_names

        # Fast Fail: if all parameters fixed -> length is fixed
        if not variable_params:
            return False

        # Fast-Path: check explicit length_param_names (if available)
        explicit_names = self.length_param_names
        if explicit_names is not None:
            # if any name in the list is variable, the length is variable
            for name in explicit_names:
                if name in variable_params:
                    return True
            # else if empty or non are variable we assume length is fixed
            return False

        # Fallback: use autograd to check sensitivity of length function
        # get the distance function + parameter values
        _, len_func, _, val_nodes = self.get_sim_functions()
        
        # check sensitivity for each variable parameter     
        for i, name in enumerate(self.param_names):
            if name in variable_params:
                # grad with argnum requests: d(len_func) / d(Param_i)
                sensitivity_func = grad(len_func, argnum=i)
                g = sensitivity_func(*self.values)
                
                if abs(g) > 1e-9:
                    return True

        return False


    @property
    def length(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        _, len_func, _ = self.get_sim_functions()
        return len_func(*self.values)

    @property
    def refractive_index(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        _, _, index_func, val_nodes = self.get_sim_functions()
        return index_func(*self.values) if index_func is not None else None

    @property
    def values(self) -> list[float | None]:
        """
        Returns the current float values of all parameters defined in get_matrix signature.
        Useful for debugging, __str__, or evaluating matrices for non-differentiable checks.
        """
        ret = []
        for name in self.param_names:
            obj = getattr(self, name)
            print(type(obj))
            
            # 1. Unwrap InputNode / Parameter / Symbol
            if isinstance(obj, ASTNode):
                ret.append(obj.value)
            # 2. Pass None through (e.g. for ABCD.n=None)
            elif obj is None:
                ret.append(None)
            # 3. Fallback: throw error 
            else:
                print()
                raise ValueError(f"Unexpected type {type(obj)} of parameter value {obj} for parameter {self}.{name}")
        return ret


    #------------------
    # PROGRAM INTERFACE
    #------------------
    def variable(self, *variable_params: str):
        """
        Fluent setter to mark a parameter as variable (e.e. trainable).
        Usage: 
            ThinLens(f=0.1).variable()    -> marks 'f'
            ABCD(...).variable()          -> marks 'A', 'B', 'C', 'D'
            ABCD(...).variable('A')       -> marks only 'A'
            ABCD(...).variable('A', 'B')  -> marks only 'A', 'B'

        """
        # no arg -> set all parameters as variable
        if not variable_params:
            variable_params = self.param_names

        # loop through all provided param names, set them as variable
        for p in variable_params:
            if p not in self.param_names:
                raise ValueError(
                    f"Parameter '{p}' not found in {self.__class__.__name__}. "
                    f"Available: {self.param_names}"
                )

            node = getattr(self, p)
            if isinstance(node, InputNode) and isinstance(node.node, Parameter): # only parameters have toggable constness
                node.fixed = False

        return self

    def fixed(self):
        """
        Fluent setter to mark all parameters as fixed (non-trainable) (default)
        """
        for p in self.variable_parameter_names:
            node = getattr(self, p)
            if isinstance(node, InputNode) and isinstance(node.node, Parameter): # only parameters have toggable constness
                node.fixed = True

        return self


    #------------
    # OVERRIDABLE
    #------------
    def init_placeholders(self, environment: "Environment"):
        pass 

    @property
    def refractive_index_param_names(self) -> list[str] | None:
        """
        Explicitly lists parameters that affect refractive index (if refractive index relevant for ABCD matrix).
        """
        return None

    @property
    def length_param_names(self) -> list[str] | None:
        """
        Explicitly lists parameters that affect physical length.
        - Return ['d', 'L', ...(any other parameters that influence the length)] to enable Fast-Path checking.
        - Return [] to explicitly declare this as a Fixed-Length/Thin element.
        - Return None (default) to fall back to use autodiff sensitivity checks.

        .. warning::
            It is recommended to override this property in subclasses for performance optimization.
        """
        return None

    @abstractmethod
    def get_sim_functions(self) -> tuple[callable, callable, callable]:
        """
        Return simulation data for the optical element.
        This should return a tuple of (matrix_function, length_function, parameter_value(s)).
          - matrix_function: A callable that returns the ABCD matrix given the parameter(s).
          - length_function: A callable that returns the effective length of the element given the parameter(s).
          - index_function:  A callable that returns the refractive index of the element given the parameter(s).

        We return functions to enable auto differentiation over parameters
        Length and refractive index require their own functions because they cannot be derived from the ABCD matrix alone.
        """
        raise NotImplementedError("Subclasses must implement get_sim_functions method.")
