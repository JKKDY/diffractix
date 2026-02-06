
import diffractix as dfx
from diffractix.graph import Node, Parameter
from diffractix.graph import ops
from diffractix.elements import *
from diffractix.beams import GaussianBeam
from diffractix.system import System
from diffractix.beams import GaussianBeam
from diffractix.composites import *
from diffractix.graph.ast import *


if __name__ == "__main__":
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
    input_beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

    sys = System()
    sys.add_input(input_beam)
    # sys.add(Space(d=0.2, ))
    # sys.add(ThinLens(f=0.1).variable())
    # sys.add(Space(d=0.3).variable())
    sys.add(ABCD(A=5, C=1).variable('A'))
    # sys.add(Space(d=0.2))
    # sys.add(ThinLens(f=0.2).variable())
    # sys.add(Space(d=0.2, n=1.3))
    # sys.add(Slab(d=0.2, n = 1.5))
    # sys.add(Space(d=0.2))
    # sys.add(Mirror(R = 3))
    # sys.add(Space(d=0.2))
    print(sys)
    sim = sys.build()

    sim.run()


 
  