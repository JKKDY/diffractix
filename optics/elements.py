import inspect
import uuid

from dataclasses import dataclass, field, InitVar
from abc import abstractmethod

import numpy as np
from .core import Entity


@dataclass
class OpticalElement(Entity):
    # Track which parameters are optimizable (e.g., {'f'}, {'d'}, {'A', 'D'})
    # By default, this is empty (everything is fixed)
    _variable_params: set[str] = field(default_factory=set, repr=False)

    @property
    def param_names(self) -> list[str]:
        """
        Returns the list of parameter names corresponding to the simulation data.
        The order mathes that of self.get_sim_data()'s returned parameters (3rd return value).

        e.g. Lens.get_matrix(self, f) -> returns ['f']
        e.g. ABCD.get_matrix(self, A, B, C, D) -> returns ['A', 'B', 'C', 'D']
        """
        sig = inspect.signature(self.get_matrix)
        return list(sig.parameters.keys())


    def variable(self, *param_names: str):
        """
        Fluent setter to mark a parameter as variable (e.e. trainable).
        Usage: 
            Lens(f=0.1).variable()        -> marks 'f'
            ABCD(...).variable()          -> marks 'A', 'B', 'C', 'D'
            ABCD(...).variable('A')       -> marks only 'A'
            ABCD(...).variable('A', 'B')  -> marks only 'A', 'B'

        """
        valid_names = self.param_names

        # no arg -> set all parameters as variable
        if not param_names:
            self._variable_params.update(valid_names)
            return self

        # loop through all provided param names, set them as variable
        for p in param_names:
            if p not in valid_names:
                raise ValueError(
                    f"Parameter '{p}' not found in {self.__class__.__name__}. "
                    f"Available: {valid_names}"
                )
            self._variable_params.add(p)

        return self

    def fixed(self):
        """
        Fluent setter to mark all parameters as fixed (non-trainable) (default)
        """
        self._variable_params.clear()
        return self

    @property
    def has_variable_length(self) -> bool:
        """
        Returns True if the physical length of this element changes when any of its currently variable parameters change.
        Returns False if length is constant regardless of variable parameters.

        The function performs sensitivity checks using autograd to determine if length depends on any variable parameter.
        """

        # shortcircuit: if all parameters fixed: length is fixed
        if not self._variable_params:
            return False

        # get the distance function + parameter values
        _, len_func, val = self.get_sim_data()
        args = val if isinstance(val, Iterable) else [val]
        
        # check sensitivity for each variable parameter     
        for i, name in enumerate(self.param_names):
            if name not in self._variable_params: 
                continue

            try:
                # Ask Autograd: "What is d(Length) / d(Param_i)?"
                # argnum=i tells it to differentiate w.r.t the i-th argument
                sensitivity_func = grad(len_func, argnum=i)
                g = sensitivity_func(*args)
                
                # If gradient is non-zero, the length depends on this variable!
                if abs(g) > 1e-9:
                    return True
                    
            except Exception:
                # If function is non-differentiable (e.g. constant), 
                # grad might fail or return 0. We assume fixed.
                pass
                
        return False

    @property
    def length(self) -> float:
        """
        Returns the physical length of this element at its current parameter values.
        """
        _, len_func, val = self.get_sim_data()
        args = val if isinstance(val, Iterable) else [val]
        return len_func(*args)

    @property
    def length_param_names(self) -> Optional[List[str]]:
        """
        Explicitly lists parameters that affect physical length.
        - Return ['d', 'L'] to enable Fast-Path checking.
        - Return [] to explicitly declare this as a Fixed-Length/Thin element.
        - Return None (default) to fall back to robust Autograd sensitivity checks.

        .. warning::
            It is recommended to override this property in subclasses for performance optimization.
        """
        return None

    @abstractmethod
    def get_matrix(self):
        """Return the ABCD matrix of the optical element."""
        raise NotImplementedError("Subclasses must implement get_matrix method.")

    @abstractmethod
    def get_sim_data(self):
        """
        Return simulation data for the optical element.
        This should return a tuple of (matrix_function, length_function, parameter_value(s)).
          - matrix_function: A callable that returns the ABCD matrix given the parameter(s).
          - length_function: A callable that returns the effective length of the element given the parameter(s).
          - parameter_value(s): The current value(s) of the parameter(s) (must be a single value or a flat iterable of values).
        We return functions to enable auto differentiation over parameters
        """
        raise NotImplementedError("Subclasses must implement get_sim_data method.")



@dataclass(kw_only=True)
class Space(OpticalElement):
    d: float
    n: float = 1.0  # Refractive index of the medium

    @property
    def length_param_names(self):
        return ['d']

    def get_matrix(self, d):
        # Note: self.n is baked into this bound method
        return np.array([[1.0, d/self.n], [0.0, 1.0]])

    def get_sim_data(self):
        # Length Logic: The physical length is exactly the parameter 'd'
        return self.get_matrix, lambda d : d, self.d


@dataclass(kw_only=True)
class ThinLens(OpticalElement):
    f: float # Focal length in meters

    @property
    def length_param_names(self):
        return []

    def get_matrix(self, f):
        return np.array([[1.0, 0.0], [-1.0/f, 1.0]])

    def get_sim_data(self):
        # Length Logic: Lenses are thin, so they add 0.0 to position
        return self.get_matrix, lambda _: 0.0, self.f





@dataclass(kw_only=True)
class ABCD(OpticalElement):
    """
    A black-box optical element defined manually by its matrix.
    Useful for subsystems or unknown optics.

    Usage:
        el = ABCD(matrix=np.eye(2))           # Option 1: Full Matrix
        el = ABCD(A=1, B=0.1, C=0, D=1)       # Option 2: Scalars
        el = ABCD()                           # Option 3: Default Identity
    """
    matrix: np.ndarray = field(default=None)

    A: InitVar[float] = None
    B: InitVar[float] = None
    C: InitVar[float] = None
    D: InitVar[float] = None

    @property
    def length_param_names(self):
        return [] # no length for now

    def __post_init__(self, A, B, C, D):
        # if matrix is explicitly provided, use it (ignores A,B,C,D)
        if self.matrix is not None:
            return

        # if matrix is missing, build it from A, B, C, D
        # default to identity if specific values are missing
        val_A = A if A is not None else 1.0
        val_B = B if B is not None else 0.0
        val_C = C if C is not None else 0.0
        val_D = D if D is not None else 1.0
        
        self.matrix = np.array([[val_A, val_B], [val_C, val_D]])

    def get_matrix(self, A, B, C, D):
        return np.array([[A, B], [C, D]])

    def get_sim_data(self):
        # Length Logic: ??? 0.0 for now
        # TODO: Allow user to specify effective length if desired
        return self.get_matrix, lambda *args: 0.0, self.matrix.flatten()

    