import diffractix as dfx

from diffractix.graph import Parameter
from diffractix.beams import GaussianBeam
from diffractix.elements import Space
from diffractix.solver import Solver
from diffractix.system import System


INITIAL_RADIUS = 100e-6
TARGET_RADIUS = 150e-6

beam = GaussianBeam.from_waist(
    w0=100e-6,
    wavelength=1064e-9,
)

distance = Parameter(0.05).variable()
space = Space(d=distance)

system = System()
system.add_input_beam(beam)
system.add(space)

solver = Solver(system)
solver.target(
    lambda ctx: (ctx.after(space).w - TARGET_RADIUS) / TARGET_RADIUS
)

solution = solver.solve()

print("Success:", solution.success)
print("Distance:", solution[distance])
print("Beam radius:", solution.run().after(space).w)
print("Original distance:", distance.value)
