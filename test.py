
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



def test_beam_expander():
    print("--- Setting up Galilean Beam Expander ---")
    sys = System()
    
    # 1. Input: 1mm waist, perfectly collimated (Plane wave)
    sys.add_input(GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9, z_waist_loc=0.0))
    
    # 2. First Lens: Diverging
    # Updated guess places the system in the telescope regime (M=5x)
    lens1 = ThinLens(f=-0.1).variable()
    lens1.f.min_val = -1.0
    lens1.f.max_val = -0.01
    
    # 3. Second Lens: Converging
    # Updated guess matches the 0.2m spacing (-0.05 + 0.25 = 0.2)
    lens2 = ThinLens(f=0.1).variable()
    lens2.f.min_val = 0.01
    lens2.f.max_val = 1.0

    sys.add(lens1)
    sys.add(Space(d=0.2))
    sys.add(lens2)
    sys.add(Space(d=0.5)) # Target z will be at 0.7m

    print(f"Initial Guess for Lens 1 f: {lens1.f.value:.4f} m")
    print(f"Initial Guess for Lens 2 f: {lens2.f.value:.4f} m")

    sim = sys.build()
    res = sim.run()
    print(sys)
    print(res)

    res.plot()

    print("\n--- Running Optimizer ---")
    opt = Optimizer(sys)
    
    # Target 1: Expand to 4mm
    opt.constrain_beam(z=0.7, w=4e-3, weight=1.0, kind='exact')
    
    # Target 2: Must be perfectly collimated. 
    opt.constrain_beam(z=0.7, R=np.inf, weight=100.0, kind='exact')

    result = opt.solve()

    print("\n--- Optimization Results ---")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Cost (Loss): {result.cost:.2e}")
    print(f"Optimized Lens 1 f: {result.x[0]:.4f} m")
    print(f"Optimized Lens 2 f: {result.x[1]:.4f} m")
    
    # Update system parameters with the optimized array results
    lens1.f.value = result.x[0]
    lens2.f.value = result.x[1]

    sim = sys.build()
    res = sim.run()
    print(sys)
    print(res)

    res.plot()


if __name__ == "__main__":
    test_beam_expander()
    
  
# if __name__ == "__main__":
#     test_optimizer()
  