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

from ..graph import Node, Parameter


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
    label: str = None 
    _counts: ClassVar[dict] = {} # ClassVar so dataclasses doesn't treat this as an instance field

    # Track which parameters are optimizable (e.g., {'f'}, {'d'}, {'A', 'D'})
    # By default, this is empty (everything is fixed)
    _variable_params: set[str] = field(default_factory=set, repr=False)


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
        # set label to *class_name*#instantaitions 
        if self.label is None:
            cls = self.__class__
            
            if cls not in OpticalElement._counts:
                OpticalElement._counts[cls] = 0
            
            OpticalElement._counts[cls] += 1
            self.label = f"{cls.__name__}{OpticalElement._counts[cls]}"


    def __setattr__(self, name, value):
        # Intercept Physics Parameters -> Wrap in ast.Parameter node
        if name in getattr(self.__class__, '_param_names', set()):

            if not isinstance(value, Node) and value is not None:
                value = Parameter(value=value, name=name, fixed=True, owner=self)
        
        super().__setattr__(name, value)


    def __str__(self):
        # Identity and Basic Geometry
        name = f"{self.__class__.__name__}"
        label_str = f" '{self.label}'" if self.label else ""
        header = f"{name:<15}{label_str:<20}L={self.length:.4g} {'[VAR]' if self.has_variable_length else '[FIX]'}"
        
        # parameters
        _, _, _, current_vals = self.get_sim_data()
        names = self.param_names
        
        param_details = []
        for name, val in zip(names, current_vals):
            # Tag variable parameters for the UI/CLI
            status = "[VAR]" if name in self._variable_params else "[FIX]"
            param_details.append(f"{name}={val:.4g} {status}")
        
        params_str = " | " + " | ".join(param_details)
      
        return f"{header:<50} {params_str}"
  


    #----
    # API
    #----
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
            self._variable_params.add(p)
            obj = getattr(self, p)
            obj.fixed = False

        return self

    def fixed(self):
        """
        Fluent setter to mark all parameters as fixed (non-trainable) (default)
        """
        self._variable_params.clear()
        for p in variable_params:
            obj = getattr(self, p)
            obj.fixed = True

        return self

    #------------------
    # @PROPERTY METHODS
    #------------------
    @property
    def param_names(self) -> list[str]:
        """
        Returns the list of parameter names corresponding to the simulation data.
        The order mathes that of self.get_sim_data()'s returned parameters (4. return value).

        e.g. Lens.get_matrix(self, f) -> returns ['f']
        e.g. ABCD.get_matrix(self, A, B, C, D) -> returns ['A', 'B', 'C', 'D']
        """
        return self.__class__._param_names

    @property
    def has_variable_length(self) -> bool:
        """
        Returns True if the physical length of this element changes when any of its currently variable parameters change.

        Strategies:
          1. Fast Fail: No parameters are variable.
          2. Explicit Fast-Path: Checks length_param_names (if provided).
          3. Implicit Fallback: autograd sensitivity check.
        """
        # Fast Fail: if all parameters fixed -> length is fixed
        if not self._variable_params:
            return False

        # Fast-Path: check explicit length_param_names (if available)
        explicit_names = self.length_param_names
        if explicit_names is not None:
            # if any name in the list is variable, the length is variable
            for name in explicit_names:
                if name in self._variable_params:
                    return True
            # else if empty or non are variable we assume length is fixed
            return False

        # Fallback: use autograd to check sensitivity of length function
        # get the distance function + parameter values
        _, len_func, _, vals = self.get_sim_data()
        
        # check sensitivity for each variable parameter     
        for i, name in enumerate(self.param_names):
            if name in self._variable_params: 
                # grad with argnum requests: d(len_func) / d(Param_i)
                sensitivity_func = grad(len_func, argnum=i)
                g = sensitivity_func(*vals)
                
                if abs(g) > 1e-9:
                    return True

        return False


    @property
    def length(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        _, len_func, _, args = self.get_sim_data()
        return len_func(*args)

    @property
    def refractive_index(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        _, _, index_func, args = self.get_sim_data()
        return index_func(*args) if index_func is not None else None


    #------------
    # OVERRIDABLE
    #------------
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
    def get_matrix(self) -> np.ndarray:
        """Return the ABCD matrix of the optical element."""
        raise NotImplementedError("Subclasses must implement get_matrix method.")

    @abstractmethod
    def get_sim_data(self) -> tuple[callable, callable, callable, Iterable[float]]:
        """
        Return simulation data for the optical element.
        This should return a tuple of (matrix_function, length_function, parameter_value(s)).
          - matrix_function: A callable that returns the ABCD matrix given the parameter(s).
          - length_function: A callable that returns the effective length of the element given the parameter(s).
          - index_function:  A callable that returns the refractive index of the element given the parameter(s).
          - parameter_value(s): The current value(s) of the parameter(s) (must be a single value or a flat iterable of values).

        We return functions to enable auto differentiation over parameters
        Length and refractive index require their own functions because they cannot be derived from the ABCD matrix alone.
        """
        raise NotImplementedError("Subclasses must implement get_sim_data method.")
