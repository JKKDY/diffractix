
from scipy.optimize import least_squares
from autograd import jacobian

from ..simulation import Simulation

import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import least_squares

class Optimizer: 
    def __init__(self, system): 
        self.system = system
        self.sim = system.build()
        
        self.constraint_funcs = []


    def constrain_beam(self, *, z: float, w: float = None, R: float = None, weight: float = 1.0, kind: str = 'exact'): 
        """Generates callable constraints and adds them to the solver."""
        if w is None and R is None:
            raise ValueError("You must specify either w or R.")

        # Register Waist Constraint
        if w is not None:
            def w_residual(hist, theta):
                idx = np.abs(hist[:, 0] - z).argmin()
                actual_w = hist[idx, 1]
                
                diff = (actual_w - w) * 1e3 # Scale to mm to prevent premature termination
                
                if kind == 'exact': err = diff
                elif kind == 'min': err = np.maximum(0.0, -diff)
                elif kind == 'max': err = np.maximum(0.0, diff)
                else: err = 0.0
                
                return err * weight
            
            self.constraint_funcs.append(w_residual)

        # Register Curvature Constraint
        if R is not None:
            def R_residual(hist, theta):
                idx = np.abs(hist[:, 0] - z).argmin()
                actual_R = hist[idx, 2]
                
                # Convert to curvature to handle R = infinity safely
                actual_curv = 1.0 / actual_R if actual_R != 0 else 0.0
                target_curv = 1.0 / R if (R is not None and not np.isinf(R) and R != 0) else 0.0
                
                diff = actual_curv - target_curv
                
                if kind == 'exact': err = diff
                elif kind == 'min': err = np.maximum(0.0, -diff)
                elif kind == 'max': err = np.maximum(0.0, diff)
                else: err = 0.0
                
                return err * weight
                
            self.constraint_funcs.append(R_residual)

        return self
        

    def constrain_parameter(self, param_node , min_val=None, max_val=None, weight=1.0):
        # param is an AST Node (e.g., lens.f)

        # look up the index in the theta vector (degrees of freedom) using object identity
        try:
            theta_idx = self.sim.variable_params.index(param_node)
        except ValueError:
            raise ValueError(f"Parameter '{param_node.name}' is fixed or not a valid degree of freedom.")
            
        # build the callable bound to that specific index
        def param_residual(hist, theta):
            val = theta[theta_idx]
            err = 0.0
            if min_val is not None:
                err += np.maximum(0.0, min_val - val)
            if max_val is not None:
                err += np.maximum(0.0, val - max_val)
            return err * weight
            
        self.constraint_funcs.append(param_residual)
        return self

    def add_constraint(self, constraint: callable):
        self.constraint_funcs.append(constraint)


    def solve(self): 
        def residuals(theta):
            hist = self.sim.run_for_optimizer(theta)
            return np.array([func(hist, theta) for func in self.constraint_funcs])

        jac = jacobian(residuals)
        
        res = least_squares(
            residuals, 
            self.sim.initial_values, 
            jac=jac, 
            method='trf' 
        )
        return res