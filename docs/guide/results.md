# Running and reading results

`simulation.run()` propagates the source through the system and returns a
`SimulationResult`: the light state before and after every element, with its position.
A result holds numbers. For the same structure as expressions, used in objectives and
constraints, see [Results as expressions](#results-as-expressions) below.

## Running

Start with a design and call `System.build()`. See [Building a simulation](building.md)
for what the build checks. The example below uses a ray and a thin lens.

```python
from diffractix import Plane, Ray, Space, System, ThinLens
from diffractix.units import mm

gap = Space(d=50 * mm, label="Gap").variable(d=(10 * mm, 100 * mm))
lens = ThinLens(f=100 * mm, label="Lens").variable(f=(50 * mm, 200 * mm))
image = Plane(label="Image")
system = System(gap, lens, Space(d=100 * mm), image, source=Ray(x=1 * mm, theta=0.0))
simulation = system.build()

result = simulation.run()
print(result.final.x)
```

```text title="Output"
0.0
```

Without arguments, `run()` uses the starting values of the variables.

## Running with other values

`simulation.initial_values` returns a copy of the starting values that you can change
by parameter. Pass it to `run()` to try a different design. You can also pass a
vector in build order, called [theta](building.md#the-variable-vector-theta).
Here `theta` refers to the design variables. On a ray, `ray.theta` is its angle
in radians.

```python
values = simulation.initial_values
values[lens.f] = 120 * mm
print(f"{simulation.run(values).final.x / mm:.4f} mm")
```

```text title="Output"
0.1667 mm
```

Changing the copy does not change the Simulation or the declared objects. Running only
evaluates the design: it does not check bounds or requirements, which is the solver's
task.

## Looking up the light

Index a result by an element to get its **stage**: the state entering it (`before`), the
state leaving it (`after`), and the positions where it starts and ends:

```python
stage = result[lens]
print(stage.before.x, stage.after.theta, stage.z_start, stage.z_end)
```

```text title="Output"
0.001 -0.01 0.05 0.05
```

A stage of an element inside a composite is looked up by that element, for example
`result[telescope.lens1]`. An element that appears more than once needs its
occurrence, counted from zero: `result[lens, 1]`.

The whole trace is also available in order:

- `result.source` and `result.final`: the first and last state;
- `result.states`: the source, then the state after each stage;
- `result.z`: the position of each of those states;
- `result.stages`: every stage in propagation order, including generated spaces.

Each state has the quantities of its kind: `x` and `theta` for rays and bundles, and
for a Gaussian beam its radius `w`, curvature radius `R`, waist radius `w0`, distance
to the waist `z0`, Rayleigh range `z_r`, Gouy phase and more. The
[states API page](../api/states.md) lists them all.

`print(result)` shows the trace as a table.

## Plotting

`result.plot()` draws the light along the system, with the elements marked, and
returns the Matplotlib figure and axes:

```python
figure, axes = result.plot()
```

It does not show or save the figure; call `matplotlib.pyplot.show()` or
`figure.savefig(...)` for that. Rays are drawn as lines and a Gaussian beam as its
radius; the beam is drawn exactly inside spaces, while thick elements such as a `GRIN`
medium show only their boundary states.

## Results as expressions

`simulation.symbolic` has the same structure as a result, but every quantity is an
expression in the variables instead of a number:

```python
sim = simulation.symbolic
height = sim[image].after.x
```

`height` is the ray height at the image as a function of `theta`. Expressions like it
are what objectives and constraints are made of, see [Solving](solving.md). They can
be combined like any other expression, for example `sim[image].z_start - sim[lens].z_end`
for a distance between two elements.
