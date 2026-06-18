
import diffractix as dfx
from diffractix.graph import Node, Parameter
from diffractix.graph import ops
from diffractix.elements import *
from diffractix.system import System
from diffractix.beam import GaussianBeam
from diffractix.composites import *
from diffractix.graph.ast import *


# if __name__ == "__main__":
    # L1 = ThinLens(f=0.1, label="L1").variable()
    # L2 = ThinLens(f=0.1, label="L2").variable()
    # S1 = Space(d=L1.f + L2.f, label="S1")
    # S2 = Space(d=2*S1.d, label="S1")

    # # S4 = Space(d=L1.f + L2.f, label="S2")
    # # S2 = Space(d=L2.f + L1.f, label="S3")
    # # S4 = Space(d=S2.d * 9 + 828 * L1.f * 2983, label="S3")
    

    # # # S2 = Space(d=2 * S1.d, label="S2")
    # # # S3 = Space(d= S2.d * L1.f, label="S3")

    # # for k, v in Node._cache.items():
    # #     print(k, v)

    # # print(S1.d.value)
    # # print(S2.d.value)
    # # print(S4.d.value)

    # # print(S1.length)

    # # S5 = Space(d = 2)
    # # S5.d = 5
    # # print(S5.length)
    # # print(type(S5.d))
    # # print(S5.d)

    # # print(S5.has_variable_length)
    # # S5.variable()
    # # print(S5.has_variable_length)
    # print(S1.length)

    # print(S1.has_variable_length)
    # L1.fixed()
    # L2.fixed()
    # print(S1.has_variable_length)
    # print(S1.length)

    # S1.variable("d")

    # print(S1.has_variable_length)
    # print(S1.length)
    # print(S2.length)
    # L1.f = 0.3
    # print(S1.length)
    # print(S2.length)
    # # S1.d = 2
    # # print(S1.length)
    # # print(S2.length)


    # input_beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

    # sys = System()
    # sys.add_input(input_beam)
    # sys.add(Space(d=0.2, ))
    # sys.add(ThinLens(f=0.1).variable())
    # sys.add(Space(d=0.3).variable())
    # sys.add(ABCD(A=5, C=1).variable('A'))
    # sys.add(Space(d=0.2))
    # sys.add(ThinLens(f=0.2).variable())
    # sys.add(Space(d=0.2, n=1.3))
    # sys.add(Slab(d=0.2, n = 1.5))
    # sys.add(Space(d=0.2))
    # sys.add(Mirror(R = 3))
    # sys.add(Space(d=0.2))
    # sim = sys.build()

    # print(sys)

    # print("Initial Parameters (Source_w0, Lens_f, Space_d):", sim.initial_params)
    # # print(sim.initial_params)

    # result = sim.run()
    # # print(json.dumps(result.export(), indent=2))

    # # expect output beam to have w = 0.001 and R = 0.1 at z = 0.4
    # plot_simulation(result)

    # L1 = ThinLens(f=0.1, label="L1").variable()
    # L2 = ThinLens(f=0.1, label="L2").variable()
    # # S1 = Space(d=L1.f + L2.f, label="S1")
    # # S4 = Space(d=L1.f + L2.f, label="S1")
    # # S2 = Space(d=2 * S1.d, label="S2")
    # # S3 = Space(d= S2.d * L1.f, label="S3")

    # for k, v in Node._cache.items():
    #     print(k, v)
    # input_beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

    # sys = System()
    # sys.add_input(input_beam)
    # sys.add(Space(d=0.2, ))
    # sys.add(ThinLens(f=0.1).variable())
    # sys.add(Space(d=0.3).variable())
    # sys.add(ABCD(A=5, C=1).variable('A'))
    # sys.add(Space(d=0.2))
    # sys.add(ThinLens(f=0.2).variable())
    # sys.add(Space(d=0.2, n=1.3))
    # sys.add(Slab(d=0.2, n = 1.5))
    # sys.add(Space(d=0.2))
    # sys.add(Mirror(R = 3))
    # sys.add(Space(d=0.2))
    # print(sys)
    # sim = sys.build()

    # sim.run()


 
import autograd.numpy as np
# Adjust these imports to match your actual diffractix folder structure
from diffractix.system import System
from diffractix.elements import Space, ThinLens
from diffractix.graph.ast import Parameter
from diffractix.beam import GaussianBeam
from diffractix.core.optimizer import Optimizer

def test_optimizer():
    print("--- Setting up Physics System ---")
    sys = System()
    
    # 1. Input: 1mm waist, 1064nm laser
    sys.add_input(GaussianBeam.from_waist(w0=5.4e-3, wavelength=1064e-9))
    
    lens = ThinLens(f=0.5).variable()

    sys.add(Space(d=0.2))
    sys.add(lens)
    sys.add(Space(d=1.8))

    print(f"Initial Guess for Lens f: {lens.f.value:.4f} m")

    sim = sys.build()
    res = sim.run()
    print(sys)
    print(res)

    res.plot()


    print("\n--- Running Optimizer ---")
    # 5. Define the Inverse Problem
    opt = Optimizer(sys)
    target_w = 50e-6 
    opt.constrain_beam(z=1.0, w=target_w, weight=1.0, kind='exact')
    result = opt.solve()


    print("\n--- Optimization Results ---")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Cost (Loss): {result.cost:.2e}")
    print(f"Optimized Lens f: { result.x[0]:.4f} m")
    
    lens.f = result.x[0]

    sim = sys.build()
    res = sim.run()
    print(sys)
    print(res)

    res.plot()

    return



def test_fiber_coupler():
    print("--- Setting up Diode-to-Fiber Coupler ---")
    sys = System()
    
    # 1. Input: 850nm Laser Diode (5um waist)
    sys.add_input(GaussianBeam.from_waist(w0=5e-6, wavelength=850e-9, z_waist_loc=0.0))
    
    # 2. Fixed space from diode to lens (20 mm)
    sys.add(Space(d=0.02, label="Diode_Gap"))
    
    # 3. Variable Thick Lens
    lens = ThickLens(d=0.01, n=1.51, R1=0.015, R2=-0.015).variable('R1', 'R2')
    # R1: Must be convex, between 5mm and 500mm
    lens[0].R.min_val = 0.005  
    lens[0].R.max_val = 0.500  
    
    # R2: Must be convex (negative), between -500mm and -5mm
    lens[2].R.min_val = -0.500 
    lens[2].R.max_val = -0.005
    sys.add(lens)
    
    # 4. The Translation Stage (Variable Space)
    # We allow the solver to slide the fiber anywhere from 10mm to 150mm away
    fiber_gap = Space(d=0.07, label="Translation_Stage").variable('d')
    fiber_gap.d.min_val = 0.01
    fiber_gap.d.max_val = 0.25
    sys.add(fiber_gap)
    
    # 5. Dummy Element for Targeting
    # This element has 0 thickness, so it just tracks the Z-plane after the variable gap
    fiber_face = Space(d=0.0, label="Fiber_Face")
    sys.add(fiber_face)

    print(f"Initial Guess R1: {lens[0].R.value * 1000:.2f} mm")
    print(f"Initial Guess R2: {lens[2].R.value * 1000:.2f} mm")
    print(f"Initial Guess Gap: {fiber_gap.d.value * 1000:.2f} mm")

    print("\n--- Running Optimizer ---")
    opt = Optimizer(sys)
    
    # Dynamic Targeting: The solver dynamically tracks 'fiber_face' 
    # even as 'fiber_gap' expands and contracts during optimization.
    opt.constrain_beam(z=fiber_face, w=40e-6, weight=1.0, kind='exact')
    opt.constrain_beam(z=fiber_face, R=np.inf, weight=1.0, kind='exact')

    result = opt.solve(True)

    print("\n--- Optimization Results ---")
    print(f"Success: {result.success}")
    print(f"Cost (Loss): {result.cost:.2e}")
    
    # Theta vector order: [R1, R2, fiber_gap.d]
    opt_R1 = result.x[0]
    opt_R2 = result.x[1]
    opt_gap = result.x[2]
    
    print(f"Optimized R1: {opt_R1 * 1000:.2f} mm")
    print(f"Optimized R2: {opt_R2 * 1000:.2f} mm")
    print(f"Optimized Gap: {opt_gap * 1000:.2f} mm")
    
    # Update AST
    lens[0].R.value = opt_R1
    lens[2].R.value = opt_R2
    fiber_gap.d.value = opt_gap

    sim = sys.build()
    final_res = sim.run()
    
    print("\nFinal State (After Optimization):")
    print(sys)
    print(final_res)
    final_res.plot()

if __name__ == "__main__":
    test_fiber_coupler()
    
  
# if __name__ == "__main__":
#     test_optimizer()
  