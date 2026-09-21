import diffractix as dfx


# ----------------
# Source
# ----------------

beam = dfx.GaussianBeam.from_waist(
    w0=100e-6,
    wavelength=1064e-9,
)


# ----------------
# Design variables
# ----------------

distance = dfx.Parameter(
    0.05,
    name="distance",
).variable(
    lower_bound=0.02,
    upper_bound=0.15,
)

focal_length = dfx.Parameter(
    0.10,
    name="focal_length",
).variable(
    lower_bound=0.05,
    upper_bound=0.20,
)


# ----------------
# Elements
# ----------------

input_space = dfx.Space(
    d=distance,
    label="Input Space",
)

lens = dfx.ThinLens(
    f=focal_length,
    label="Focus Lens",
)

output_space = dfx.Space(
    d=2 * distance,
    label="Output Space",
)

target = dfx.Plane(
    label="Target",
)


# Element requirement
lens.require(
    focal_length >= 0.06,
)


# ----------------
# System
# ----------------

system = dfx.System()

system.add_input_beam(beam)

system.add(input_space)
system.add(lens)
system.add(output_space)

# Explicit absolute placement
system.add(target, z=0.25)


# System requirement
system.require(
    distance >= 0.03,
)


# ----------------
# Inspect blueprint
# ----------------

print(system)


# ----------------
# Compile and inspect
# ----------------

simulation = system.build()

print()
print(simulation)