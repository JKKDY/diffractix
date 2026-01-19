
import diffractix as dfx
from diffractix.graph import Node, Parameter
from diffractix.graph import ops
from diffractix.elements import *
from diffractix.beams import GaussianBeam
from diffractix.system import System
from diffractix.beams import GaussianBeam
from diffractix.composites import *
from diffractix.graph.ast import BinaryOp, UnaryOp


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

    input_beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1064e-9)

    # 1. Initialize System
    sys = System()
    sys.add_input(input_beam)

    # --- SECTION 1: Relative/Sequential ---
    sys.add(Space(d=0.2, label="FixedEntry"))
    sys.add(ThinLens(f=0.1, label="L1").variable()) 

    # --- SECTION 2: The "Loose" Chain ---
    sys.add(Space(d=0.3, label="Slider").variable()) 
    sys.add(ABCD(A=5, C=1, label="BlackBox").variable('A'))

    # --- SECTION 3: The Anchor ---
    # Target Z = 1.5m
    sys.add(ThinLens(f=0.2, label="L2").variable(), z=1.5)

    # --- SECTION 4: Final Anchor ---
    # Target Z = 2.0m
    sys.add(Space(d=0.1, label="Detector"), z=2.0)

    print("--- BLUEPRINT ---")
    # print(sys)

    # 2. Build the Simulation
    sim = sys.build()

    print("--- COMPILED (RESOLVED) ---")
    print(sys)

    # 3. Structural Info
    print(f"Total Constraints Generated: {len(sim.constraints)}")

    # 4. DIRECT CONSTRAINT VERIFICATION
    print("\n--- Manual Constraint Verification ---")

    # Execute the math-heavy pass
    # data shape: (N_steps + 1, 3) -> [z, w, R]
    print(sim.__dict__.keys())
    data = sim.run_for_optimizer()

    # We expect L2 to be at index 5 (Input + FixedEntry + L1 + Slider + BlackBox + AutoSpace)
    # We expect Detector to be at index 7 (Above + L2 + AutoSpace)
    anchors = [
        {"label": "L2 Anchor", "index": 5, "target": 1.5},
        {"label": "Detector Anchor", "index": 7, "target": 2.0}
    ]

    all_passed = True
    for anchor in anchors:
        actual_z = data[anchor['index'], 0]
        residual = (actual_z - anchor['target']) * 1e3 # convert to mm
        
        status = "✅" if abs(residual) < 1e-7 else "❌"
        print(f"{status} {anchor['label']}: Actual Z = {actual_z:.4f}m | Target = {anchor['target']:.4f}m | Residual = {residual:.8e} mm")
        
        if abs(residual) > 1e-7:
            all_passed = False

    # 5. EXECUTE GENERATED CONSTRAINTS
    # This tests if the lambda functions we built actually work
    print("\n--- Lambda Constraint Verification ---")
    for i, const_func in enumerate(sim.constraints):
        res = const_func(sim.initial_params, data)
        print(f"Lambda {i} Output: {res:.8f} mm")

    # 6. Run Physics & Plot
    result = sim.run()
    plot_simulation(result)

    if all_passed:
        print("\nVERIFICATION COMPLETE: Layout logic and constraint generation are mathematically sound.")