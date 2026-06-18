
from scipy.optimize import least_squares
from autograd import jacobian

from ..simulation import Simulation
from ..beam import GaussianBeam 
from ..elements import OpticalElement

import autograd.numpy as np

class Optimizer: 
    def __init__(self, system): 
        self.system = system
        self.sim = system.build()
        self.constraint_funcs = []

    def constrain_beam(self, *, z: float, w: float = None, R: float = None, weight: float = 1.0, kind: str = 'exact'): 
        """Generates a callable constraint that interpolates to the exact z position."""
        if w is None and R is None:
            raise ValueError("You must specify either w or R.")

        # parse beam constraint location
        target_element = None
        offset = 0.0

        if isinstance(z, OpticalElement):
            target_element = z
        elif isinstance(z, tuple):
            target_element = z[0]
            offset = z[1]
        elif isinstance(z, (int, float)):
            offset = float(z) # Absolute z constraint
        else:
            raise TypeError("z must be a float, OpticalElement, or element + offset")

        # resolve index in compiled system
        element_idx = None
        if target_element is not None:
            try:
                element_idx = self.system._compiled_elements.index(target_element)
            except ValueError:
                raise ValueError(f"Element {target_element.label} not found in compiled system.")

        # Helper to find the correct state and propagate
        def get_interpolated_q_and_n(zs, qs, ns):
            # Calculate absolute Z for this specific solver iteration
            if element_idx is not None:
                target_z = zs[element_idx] + offset
            else:
                target_z = offset 

            # Find the closest upstream element to interpolate from
            valid_indices = np.where(zs <= target_z)[0]
            idx = valid_indices[-1] if len(valid_indices) > 0 else 0
            
            z_base = zs[idx]
            q_base = qs[idx]
            n_base = ns[idx]
            
            dz = target_z - z_base
            return q_base + dz, n_base

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
            self.add_constraint(w_residual)

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
            self.add_constraint(R_residual)

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
            
        self.add_constraint(param_residual)
        return self

    def constrain_path_waist(self, min_w: float, weight: float = 1.0):
        """
        Path Constraint: Prevents the beam from focusing tighter than min_w 
        anywhere in the physical layout.
        """
        # Pre-allocate fractional steps to ensure Autograd compatibility
        fractions = np.linspace(0, 1.0, 1000) 
        
        def path_residual(zs, qs, ns, wvl, theta):
            penalty = 0.0
            for i in range(len(zs) - 1):
                L = zs[i+1] - zs[i]
                if L <= 1e-6: # Skip thin elements like lenses
                    continue
                
                # 1. Propagate q to 10 points along this specific segment
                dz = fractions * L
                q_z = qs[i] + dz
                
                # 2. Extract w for all 10 points
                # np.minimum protects against divide-by-zero on perfectly collimated beams
                inv_q_imag = np.minimum(np.imag(1.0 / q_z), -1e-16)
                w_vals = np.sqrt(-wvl / (np.pi * ns[i] * inv_q_imag))
                
                # 3. Penalize any point where w is less than the minimum allowed waist
                violations = np.maximum(0.0, min_w - w_vals)
                penalty += np.sum(violations)
                
            return penalty * weight * 1e3 # Scale to mm for healthy gradients
            
        self.add_constraint(path_residual)
        return self

    def add_constraint(self, constraint: callable):
        self.constraint_funcs.append(constraint)


    def _extract_bounds(self):
        """Builds the 2-tuple of arrays required by scipy: ([lower...], [upper...])"""
        lower_bounds = []
        upper_bounds = []
        
        for param in self.sim.variable_params:
            lower_bounds.append(param.min_val)
            upper_bounds.append(param.max_val)
            
        return lower_bounds, upper_bounds

    def solve(self, global_search: bool = False): 
        def residuals(theta):
            # Run simulation and unpack the complex state
            zs, qs, ns, wvl = self.sim.run_for_optimizer(theta)
            # Pass all required args to the callables
            return np.array([func(zs, qs, ns, wvl, theta) for func in self.constraint_funcs])

        bounds = self._extract_bounds()
        if not global_search:
            from autograd import jacobian
            from scipy.optimize import least_squares


            jac = jacobian(residuals)
        
            res = least_squares(
                residuals, 
                self.sim.initial_values, 
                jac=jac, 
                bounds=bounds,
                method='trf' 
            )
            return res

        else:
            from scipy.optimize import differential_evolution

            # DE requires a scalar objective function (Sum of Squared Errors)
            def scalar_cost(theta):
                res_vec = residuals(theta)
                return np.sum(res_vec**2)

            # Sanitize bounds: DE cannot randomly sample across infinite space
            de_bounds = []
            for mn, mx in zip(bounds[0], bounds[1]):
                safe_min = -1e4 if np.isinf(mn) else mn
                safe_max =  1e4 if np.isinf(mx) else mx
                de_bounds.append((safe_min, safe_max))

            # Run the evolutionary algorithm
            res = differential_evolution(
                scalar_cost, 
                bounds=de_bounds,
                strategy='best1bin',
                maxiter=1000,
                popsize=15,       # 15 mutated clones per variable per generation
                mutation=(0.5, 1.0), 
                recombination=0.7,
                tol=1e-6
            )
            
            # Map .fun to .cost so the return object matches least_squares API
            res.cost = res.fun 
            return res