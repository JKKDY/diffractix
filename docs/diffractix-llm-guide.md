# Diffractix API guide for generating optical simulations

Use this guide when writing code with the current Diffractix API. Prefer public
interfaces, small models, explicit units and physically meaningful parameter bounds.
Do not invent arguments, properties or optical components. If the requested
physics is outside paraxial ABCD optics, say so rather than implying support.

## Model and conventions

Diffractix models paraxial optical systems using ABCD matrices. Lengths, focal
lengths, radii, positions, beam widths and vacuum wavelengths use metres. Angles
are in radians. A ray vector is `[x, theta]`, where theta is the physical angle,
not `n * theta`. Mirrors use the unfolded optical-path convention.

The normal workflow is:

1. Declare an optical input and an ordered sequence of elements.
2. Mark selected real scalar parameters variable and choose bounds.
3. Build a `System` into a `Simulation`.
4. Run it, inspect a `SimulationResult`, or optimize with `Solver`.
5. Check `solution.success` and `solution.feasible` before using a solution.

A System currently supports one optical input. A Simulation snapshots compiled
expressions and input values at build time; later edits to the declarative model
do not update it. Rebuild the System to include those changes. Python callable
captures remain the caller's responsibility. Bindings containing mutable graph
objects have a documented shallow-snapshot limitation.

## Imports and inputs

```python
import numpy as np
from diffractix import (
    ABCD, Backend, Constraint, GaussianAperture, GaussianBeam, GRIN,
    Interface, Mirror, Objective, Parameter, Plane, Ray, RayBundle,
    Slab, Solver, Space, System, ThickLens, ThinLens,
)
```

```python
beam = GaussianBeam.from_waist(w0=0.5e-3, wavelength=1064e-9)
# Or GaussianBeam(q=0.02j, wavelength=1064e-9, n=1.0).
ray = Ray(x=1e-3, theta=0.01, n=1.0)
bundle = RayBundle(x=np.array([-1e-3, 0.0, 1e-3]),
                   theta=np.array([0.0, 0.0, 0.0]), n=1.0)
```

Gaussian `q` must be finite with a positive imaginary part. `w` is beam radius,
not diameter. `GaussianBeam.from_w_and_R(w, R, wavelength, n=1.0)`,
`from_divergence(theta, wavelength, n=1.0)` and `from_fiber_tip(wavelength,
NA=None, MFD=None, n=1.0)` are also available. `RayBundle.parallel(radius,
count, n=1.0)` and `RayBundle.converging(radius, count, focal_distance, n=1.0)`
construct bundles. Bundle positions/angles are matching one-dimensional arrays.

The public input objects are declarative: graph-backed fields are handles, not
ordinary floats. Pass a declarative input to System, not `input.numeric()`.
Numerical states from `simulation.run()` or `solution.run()` expose numeric
values. `.numeric(bindings=...)` is for standalone input calculations.
Complex Gaussian `q` and ray-bundle arrays are numerical fields, not variable
Parameters; Gaussian wavelength and refractive index, Ray x/theta/n, and
RayBundle n can be graph-backed.

## Elements and systems

```python
space = Space(d=0.05)                 # n=None inherits the incoming medium
lens = ThinLens(f=0.1, label="Lens")
plane = Plane(label="Image")
system = System(space, lens, Space(d=0.1), plane, optical_input=ray)
simulation = system.build()
result = simulation.run()
print(result.final.x)
print(result[lens].before.x, result[lens].after.theta)
```

Common elements: `Space(d, n=None)`, `ThinLens(f)`, `Plane()`,
`Mirror(R=np.inf)`, `Interface(n1, n2, R=np.inf)`, `GRIN(d, g, n)`,
`GaussianAperture(a)`, and `ABCD(A=1, B=0, C=0, D=1, thickness=0, n=None)`.
Use keyword arguments for clarity. Elements accept `label=`. Composites include
`Slab(d, n)`, `ThickLens(d, n, R1, R2)`, `Telescope(f1, f2)` and `FourF(f1, f2)`.
A composite is flattened into optical stages during build; its child elements
remain available for result lookup.

A source's refractive index establishes its initial medium; `n=None` inherits
`System(ambient_n=...)`, default 1. An element without an explicit index inherits
the incoming medium. `Interface` performs a refractive transition; a GRIN rod in
a different medium needs suitable entrance and exit interfaces. Explicit ordinary
element indices must match the upstream medium. Fixed mismatches fail at build;
variable continuity is enforced as a constraint.

For absolute positions use `system.add(element, z=...)`. Build inserts a Space
for the gap. The default `z` refers to the front. Use `at="center"`, `at="back"`,
a length from the front, or a uniquely occurring child element when placing a
composite. `at` requires `z`. Negative generated gaps are invalid. Sequential
construction with explicit `Space` is usually simpler.

## Parameters, expressions and guesses

```python
distance = Parameter(0.05, name="distance").variable(bounds=(0.01, 0.2))
space = Space(d=2 * distance)  # dependent expression; only distance enters theta
lens = ThinLens(f=0.12)
lens.f.variable(bounds=(0.05, 0.3))
```

A Parameter is fixed unless marked `.variable()`. `.fixed()` reverses this.
Bounds are always a `(lower, upper)` pair: `variable(bounds=...)`, the
`Parameter(..., bounds=...)` constructor or `parameter.bounds = ...`; a `None`
endpoint leaves the current one unchanged. Choose an initial value inside the
bounds. A shared Parameter has one theta entry. Graph arithmetic combines Nodes;
comparisons such as `distance >= 0.01` create symbolic constraints. Do not use
symbolic expressions in Python `if` or convert them to bool.

`simulation.initial_values` returns an editable copy:

```python
simulation = System(space, lens, plane, optical_input=ray).build()
guess = simulation.initial_values
guess[distance] = 0.08
new_result = simulation.run(theta=guess.values)
```

Only variable model Parameters or direct handles can index this object. Its
`.values` vector follows the Simulation's canonical theta ordering. Edits do not
change the model or Simulation defaults. Do not assume a particular theta order.

## Result access

```python
result = simulation.run()             # default theta
result.initial                         # numerical input state
result.final                           # numerical output state
result[lens].before                    # state entering lens
result[lens].after                     # state leaving lens
result[lens].z_start, result[lens].z_end
result.z                               # all boundary z positions
result.states                          # input and each outgoing state
result.stages[0].after                 # numerical stage indexing
```

For a repeated element, use `result[element, 0]`, `result[element, 1]`, etc.,
or `result.stage(element, occurrence=1)`. Omitting the occurrence is an error
when the same object appears more than once. Element lookup uses object identity.
Gaussian states expose `w`, `R`, `q`, `z0`, `z_r`, `w0`, `gouy_phase` and `NA`;
ray states expose `x`, `theta` and `n`; bundles expose arrays `x`, `theta` and
`vectors`. `z_start`/`z_end` are stage boundary positions, not arbitrary-z probes.

## Objectives and constraints

`simulation.symbolic` provides graph expressions for result quantities. It is
bound to that particular Simulation, so use the same Simulation in Solver.

```python
from diffractix import Constraint, Objective, Solver

symbolic = simulation.symbolic
solver = Solver(simulation)
solver.target(symbolic.final.x)  # residual: minimize x**2
solver.target(value=symbolic[lens].after.x, target=0.002, scale=0.002,
              label="Lens height")
solver.target(Objective(lens.f - 0.1, weight=0.5, label="Preferred focal length"))
solver.require(lens.f >= 0.05)
solver.require(symbolic[plane].after.x <= 0.001, label="Maximum image position")
```

`target(*residuals)` minimizes the sum of weighted squared residual entries.
`Objective(expression_or_callable, weight=1.0, label=None)` sets a weight and
label. `target(value=..., target=..., scale=..., label=...)` creates the residual
`(value-target)/scale`. If scale is omitted, it defaults elementwise to
`abs(target)` for nonzero targets and 1 for zero targets. Explicit nonzero scale
is recommended for a zero target. Numerical array target and scale values are
supported with broadcasting; every scale entry must be nonzero.

`require(comparison)` accepts a graph comparison; a label can name one comparison.
For a bounded callable or two-sided constraint, use
`Constraint(expression_or_callable, lower_bound=..., upper_bound=..., label=...)`.
Bounds are inclusive; omitted bounds are infinite. Requirements can also be
attached with `system.require(...)` and `element.require(...)` before build.
System and element constraints are collected before Solver constraints. An
unsatisfied variable constraint can be solved; an unsatisfiable constant fails.
Variable Parameters used only by ordinary constraints cannot extend theta: they
must already occur in the optical model. Avoid attaching the same constraint twice.

A callable receives a result view: `lambda result: result[lens].after.w - target`.
Traceable callables become graph programs. Python branches depending on result
values may fall back to numerical evaluation; a deferred callable must return a
number or array, not a Node. Dynamic Python branches generally cannot be JIT
compiled. Use `solver.solve(jit=False)` for those; do not expect JIT to silently
fall back. Exceptions from user callables are not ignored.

## Complete inverse-design example

```python
from diffractix import Plane, Ray, Solver, System, ThinLens

lens = ThinLens(f=0.2, label="Focus")
lens.f.variable(bounds=(0.05, 0.5))
image = Plane(label="Image")
system = System(optical_input=Ray(x=0.01, theta=0.0))
system.add(lens, z=0.1)
system.add(image, z=0.2)
simulation = system.build()
sim = simulation.symbolic

solver = Solver(simulation)
solver.target(value=sim[image].after.x, target=0.0, scale=0.01,
              label="Image position")
solution = solver.solve(method="SLSQP", options={"ftol": 1e-10})
if not solution.success or not solution.feasible:
    raise RuntimeError(solution.message)
print("focal length [m]:", solution.value(lens.f))  # approximately 0.1
print("image position [m]:", solution.run()[image].after.x)  # approximately 0
```

`solve()` takes keyword-only arguments. `optimizer` is an `Optimizer` or its name (`"scipy"`, `"ipopt"`, `"nlopt"`); it defaults to SciPy with method SLSQP and `jit=True`; pass
`jit=False` for eager evaluation. Methods `"SLSQP"` and `"trust-constr"` are
supported by SciPy. `"ipopt"` and `"nlopt"` need optional packages.
Pass native options with `options={...}`. Optional `x0`, `x_lower`, `x_upper`
use the Simulation's canonical theta order and apply to that solve only.
`feasibility_tolerance` controls reported feasibility. Bounds and initial values
must have the correct shape; initial values must be finite and within the bounds.

A Solver's model becomes fixed once it has prepared callbacks. Add objectives and
constraints before its first solve; create a new Solver for another mathematical
problem. Repeated solves can use different initial vectors and bounds.

`solution.x` is the read-only optimized vector; `cost`, `success`, `message`,
`feasible`, `iterations`, `function_evaluations`, `objectives`, `constraints` and
`violations` provide reporting. Success is distinct from feasibility. Reporting
includes full model constraints and violations include parameter bounds.
`solution.value(parameter_or_expression)` reads solved design quantities;
`solution.run()` returns the optimized numerical result. `solution.simulation`
is a separate Simulation with solved defaults and current solve bounds. It can
be given to a new Solver. The declarative lens and other model objects are not
modified by optimization.

## Plotting and backends

```python
fig, ax = solution.run().plot()
# Optionally: import matplotlib.pyplot as plt; plt.show()
```

`result.plot()` and `diffractix.plotting.plot(result)` return `(figure, axes)`;
they do not show or save a figure. Plot z is in metres and transverse positions
are in millimetres. Continuous sampling is provided for Spaces; other thick
elements show boundary states rather than invented interior trajectories.

Autograd is required; JAX is optional. Select before the first numerical
operation with `DIFFRACTIX_BACKEND=autograd` or `jax`, or:

```python
from diffractix import backend
backend.configure("autograd")  # call before using numerical operations
```

For differentiable custom numerical functions use `diffractix.backend.xp`,
`grad`, `jacobian`, `hessian` and `jit`. JAX traced predicates cannot raise
ordinary Python errors based on unknown runtime values; validate concrete inputs
at the boundary. Consult `docs/numerical_backends.md` for details.

## Avoid removed or unsupported APIs

- Use `System(optical_input=...)`, not `System(input=...)`.
- Use `result[element].after`, not `result.at(element)` or `result.after(element)`.
- Use `solution.value(lens.f)`, not `solution[lens.f]` or `solution.value_of(...)`.
- Use `solver.solve(jit=False)`, not the former `use_jit` keyword.
- Do not use `Simulation.require()`; use System/element requirements or Solver.
- Do not place Python control flow on Nodes or assume interior-z probing exists.
- Do not invent array-valued or complex Parameters; use numerical state arrays.
- Do not treat changes to the original System after build as updates to Simulation.
