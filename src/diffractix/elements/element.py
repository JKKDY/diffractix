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
from functools import cached_property

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
        func = getattr(cls, "compute_matrix", None)
        if func is None:
            raise TypeError("Subclass must define compute_matrix")

        sig = inspect.signature(func)
        cls._param_names = [name for name in sig.parameters if name != "self"]

        # ensure that the base class also has the members sepecified in the signature 
        # (e.g. if compute_matrix(self, a, b) -> self.a, self.b)
        annotations = getattr(cls, '__annotations__', {})
        
        for name, type_hint in annotations.items():
            # Rough check: does the type hint string look like a Node?
            # e.g. "float | Node", "Parameter", "InputNode"
            type_hint = str(type_hint)
            keywords = ("Node", "Parameter", "InputNode", "Symbol", "Constant")
            if any(k in type_hint for k in keywords):
                if name not in cls._param_names:
                    cls._param_names.append(name)

        # validate that all parameter names in the signature have a corresponding member variable 
        # (e.g. def compute_matrix(self, a, b) -> self.a, self.b)
        for name in cls._param_names:
            if not (hasattr(cls, name) or name in annotations):
                 raise TypeError(f"Parameter '{name}' found in signature/hints but missing from fields.")



    def __post_init__(self):
        self.id = next(self._id_counter)

        # set label to *class_name*#instantaitions if no label was provided
        if self.label is None:
            cls = self.__class__
            
            if cls not in OpticalElement._counts:
                OpticalElement._counts[cls] = 0
            
            OpticalElement._counts[cls] += 1
            self.label = f"{cls.__name__}{OpticalElement._counts[cls]}"


    def __setattr__(self, name, value):
        """ 
        intercept attribute assignments to parameters; they must be wraped into AST nodes
        """

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

        # Handle Initialization (First time)
        # This occurs during __init__ or dataclass default assignment
        if isinstance(value, Node) and not isinstance(value, InputNode):
            # Already a node (e.g. Lens(f=Symbol("x")))
            new_handle = InputNode(value)
        elif convertable_to_float(value):
            # Raw number (e.g. Lens(f=0.1))
            new_node = Parameter(value=value, name=name, fixed=True, owner=self)
            new_handle = InputNode(new_node)
        elif value is None:
            new_handle = InputNode(None) # uninitialized, must be set later 
        else:
            raise TypeError(f"Initial value for {self.label}.{name} must be Node or float, got {type(value)}")


        # Handle Update (Handle already exists)
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

        
        super().__setattr__(name, new_handle)



    def __str__(self):
        # Identity and length
        name = f"{self.__class__.__name__}"
        label_str = f" '{self.label}'" if self.label else ""
        header = f"{name:<15}{label_str:<20}L={self.length:.4g} {'[VAR]' if not self.element_length.is_constant else '[FIX]'}"
        
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
            if obj is not None and hasattr(obj, "is_constant") and obj.is_constant is False:
                names.append(name)
        return names

    @property
    def param_names(self) -> list[str]:
        """
        Returns the list of parameter names corresponding to the signature of self.compute_matrix.

        e.g. Lens.compute_matrix(self, f) -> returns ['f']
        e.g. ABCD.compute_matrix(self, A, B, C, D) -> returns ['A', 'B', 'C', 'D']
        """
        return self.__class__._param_names

    @property
    def parameters(self) -> list[ASTNode]:
        """
        Returns the list of parameter nodes corresponding to the signature of self.compute_matrix.
        """
        return [getattr(self, name) for name in self.param_names]

    @property
    def length(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        return self.element_length.value

    @property
    def refractive_index(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        return self.element_refractive_index.value

    @property
    def values(self) -> list[float | None]:
        """
        Returns the current float values of all parameters defined in calculate_matrix signature.
        Useful for debugging, __str__, or evaluating matrices for non-differentiable checks.
        """
        ret = []
        for name in self.param_names:
            obj = getattr(self, name)
            
            # if InputNode / Parameter / Symbol, get its value
            if isinstance(obj, ASTNode):
                try:
                    ret.append(obj.value)
                except Exception as e:
                    ret.append(None) # if value can't be evaluated (e.g. unbound Symbol), return None          
            # Fallback: throw error 
            else:
                raise ValueError(f"Unexpected type {type(obj)} of parameter value {obj} for parameter {self}.{name}")
        return ret


    #---------------------
    # FUNCTIONAL INTERFACE
    #---------------------
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


    @cached_property
    def element_refractive_index(self) -> Node:
        """
        Returns the AST Node representing the OUTPUT refractive index.
        """
        return InputNode(None)


    #-----------------
    # ABSTRACT METHODS
    #-----------------
    @property
    @abstractmethod
    def element_length(self) -> Node:
        """Returns the AST Node representing physical length."""
        pass
        
    @abstractmethod
    def compute_matrix(self, **kwargs) -> np.ndarray:
        """
        Returns the ABCD matrix given scalar inputs.
        For each argument in the signature there must be a corresponding member variable 
        with the same name (e.g. def get_matrix(self, a, b) -> self.a, self.b).
        """
        pass
