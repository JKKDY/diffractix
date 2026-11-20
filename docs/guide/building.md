# Building a simulation

`system.build()` turns a declared design into a `Simulation`. The build checks the
design, resolves everything that is known before any optimization, and compiles the
rest into a function of the variable parameters. Running and solving then start from
the Simulation and never repeat this work.

## What the build does

In order, the build:

1. checks that the system has a source and valid elements;
2. replaces composites by their parts and resolves absolute positions, inserting a
   `Space` for each gap;
3. follows the refractive index along the path, and checks that each element matches
   the medium it is in;
4. collects the variable parameters into the vector `theta`;
5. compiles the propagation and the requirements, replacing fixed parameters by
   constants and the system's symbols by their values;
6. checks the requirements that no longer depend on a variable, and drops the ones
   that hold.

Any problem found along the way raises an error that names the element, placement or
requirement involved, and for a placement the line of code that added it.

## The variable vector theta

`theta` is the vector of all variable parameters, one entry for each. A parameter used
in several places has one entry, and an expression has none, since it is computed from
its parameters. `theta` is the only input of the simulation, and what the optimizer
changes.

`simulation.parameter_data` describes the entries in their order, with each one's
name, starting value and bounds:

```python
from diffractix import Plane, Ray, Space, System, ThinLens
from diffractix.units import mm

gap = Space(d=50 * mm, label="Gap").variable(d=(10 * mm, 100 * mm))
lens = ThinLens(f=100 * mm, label="Lens").variable(f=(50 * mm, 200 * mm))
image = Plane(label="Image")
system = System(gap, lens, Space(d=100 * mm), image, source=Ray(x=1 * mm, theta=0.0))

simulation = system.build()
for entry in simulation.parameter_data:
    print(entry.parameter_index, entry.name, entry.value, entry.lower_bound, entry.upper_bound)
```

```text title="Output"
0 d 0.05 0.01 0.1
1 f 0.1 0.05 0.2
```

Do not rely on a particular order. To set values, use `simulation.initial_values`,
which is indexed by parameter instead of position; see
[Running and reading results](results.md#running-with-other-values).

## The Simulation is a snapshot

The build copies the current values, bounds and structure of the design. Changing the
declared objects afterwards does not change an existing Simulation:

```python
lens.f.value = 150 * mm
print(simulation.parameter_data[1].value, system.build().parameter_data[1].value)
```

```text title="Output"
0.1 0.15
```

Build again after changing the design. A build is quick, and each Simulation is
independent of the others.

## Printing a simulation

`print(simulation)` shows the resolved layout: each stage with its absolute position,
type, label, length and refractive index, including the generated spaces, followed by
the variables and the requirements that remain for the optimizer.
