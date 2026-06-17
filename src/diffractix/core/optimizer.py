
from scipy.optimize import least_squares
from autograd import jacobian

from ..simulation import Simulation
from ..beam import GaussianBeam 

import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import least_squares

class Optimizer: 
    def __init__(self, system): 
        self.system = system
        self.sim = system.build()
        self.constraint_funcs = []

    def constrain_beam(self, *, z: float, w: float = None, R: float = None, weight: float = 1.0, kind: str = 'exact'): 
        """Generates a callable constraint that interpolates to the exact z position."""
        if w is None and R is None:
            raise ValueError("You must specify either w or R.")

        # Helper to find the correct state and propagate
        def get_interpolated_q_and_n(zs, qs, ns):
            # Find the last element that is BEFORE or AT our target z
            # np.where returns indices where the condition is true
            valid_indices = np.where(zs <= z)[0]
            if len(valid_indices) == 0:
                idx = 0 # Fallback if z is negative/before start
            else:
                idx = valid_indices[-1]
            
            z_start = zs[idx]
            q_start = qs[idx]
            n_start = ns[idx]
            
            # Analytical free-space propagation to exact target z
            dz = z - z_start
            q_target = q_start + dz
            
            return q_target, n_start

        if w is not None:
            def w_residual(zs, qs, ns, wavelength, theta):
                q_target, n_start = get_interpolated_q_and_n(zs, qs, ns)
                
                # Extract actual w at the target location
                actual_w, _ = GaussianBeam.w_R_from_q(q_target, wavelength, n_start)
                
                diff = (actual_w - w) * 1e3 # Scale to mm
                
                if kind == 'exact': err = diff
                elif kind == 'min': err = np.maximum(0.0, -diff)
                elif kind == 'max': err = np.maximum(0.0, diff)
                else: err = 0.0
                
                return err * weight
            self.constraint_funcs.append(w_residual)

        if R is not None:
            def R_residual(zs, qs, ns, wavelength, theta):
                q_target, n_start = get_interpolated_q_and_n(zs, qs, ns)
                
                # We extract the complex q to get curvature directly
                inv_q = 1.0 / q_target
                actual_curv = np.real(inv_q)
                target_curv = 1.0 / R if (R is not None and not np.isinf(R) and R != 0) else 0.0
                
                diff = actual_curv - target_curv
                
                if kind == 'exact': err = diff
                elif kind == 'min': err = np.maximum(0.0, -diff)
                elif kind == 'max': err = np.maximum(0.0, diff)
                else: err = 0.0
                
                return err * weight
            self.constraint_funcs.append(R_residual)

        return self
        
    def constrain_parameter(self, param_node, min_val=None, max_val=None, weight=1.0):
        try:
            theta_idx = self.sim.variable_params.index(param_node)
        except ValueError:
            raise ValueError(f"Parameter '{param_node.name}' is fixed or not a valid degree of freedom.")
            
        def param_residual(zs, qs, ns, wavelength, theta):
            val = theta[theta_idx]
            err = 0.0
            if min_val is not None: err += np.maximum(0.0, min_val - val)
            if max_val is not None: err += np.maximum(0.0, val - max_val)
            return err * weight
            
        self.constraint_funcs.append(param_residual)
        return self

    def add_constraint(self, constraint: callable):
        self.constraint_funcs.append(constraint)

    def solve(self): 
        def residuals(theta):
            # Run simulation and unpack the complex state
            zs, qs, ns, wvl = self.sim.run_for_optimizer(theta)
            # Pass all required args to the callables
            return np.array([func(zs, qs, ns, wvl, theta) for func in self.constraint_funcs])

        jac = jacobian(residuals)
        
        res = least_squares(
            residuals, 
            self.sim.initial_values, 
            jac=jac, 
            method='trf' 
        )
        return res