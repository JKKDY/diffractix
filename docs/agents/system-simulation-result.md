# From an optical system to a result

`System` holds the optical design. `build()` converts that design into a
`Simulation`. `run(theta)` evaluates the Simulation and returns a
`SimulationResult` for that parameter vector.

```text
Inputs + elements + bindings + constraints
                    |
                    v
                  System
                    | build()
                    v
                Simulation
                    | run(theta)
                    v
             SimulationResult
```

This describes the current rewritten API. Compilation, propagation and small
real optimization pipelines have integration tests under Autograd and JAX. See
[QA status](qa-status.md) for current results and remaining release work.

## Define the design

Inputs describe optical states, such as a Gaussian beam or a ray bundle.
Elements describe transformations, such as propagation through a Space or Lens.
Composites group elements and can contain other composites.

Declare graph-backed fields with a `Node` annotation or the dataclass helper
`parameter(...)`. Each field has an `InputNode` handle. Assigning a real number
creates a Parameter inside it. Assigning an expression stores the expression.
Other fields, such as arrays and complex beam parameters, stay numerical.

```python
source = GaussianBeam(q=0.02j, wavelength=1.064e-6)
lens = ThinLens(f=0.1, label="Focus").variable("f")

system = System(
    Space(d=0.05),
    lens,
    Space(d=0.1),
    optical_input=source,
)
simulation = system.build()
```

Element order defines the optical path. An absolute placement `z` requests a
position; build inserts a Space for the gap. By default `z` is where the element's
front goes. `add(element, z=..., at=...)` chooses another reference point:
`"front"`, `"center"` or `"back"`; an element inside the added composite, meaning
its front; or any length measured from the front, for example
`at=lens.element_length / 3.5`. The middle of an inner element sits at `z0` with
`add(composite, z=z0 - inner.element_length / 2, at=inner)`. An element that appears
twice in the composite is an ambiguous reference and raises ValueError, as does
using the added element itself. `at` requires `z`.
`add` converts `z` and `at` to the position of the front, which is all the
placement stores. Positions are exact: a front or a generated gap that comes out
negative, even by rounding (a station at 0.3 after gaps of 0.1 and 0.2), is
rejected or infeasible, so leave a little real room or use exactly representable
values. Inputs, elements and composites
can contribute Symbol bindings through their `bindings` properties. Use
`system.bind(key, value)` for explicit System bindings. Duplicate keys fail.

## Build the Simulation

Build performs these steps:

1. Validate the declared elements and require one optical input.
2. Collect bindings from the input, elements, composites and System.
3. Flatten composites. Retain each leaf's path, label and identity.
4. Resolve absolute placements and insert gap Spaces.
5. Resolve refractive indices and generate continuity constraints.
6. Compile the optical expressions and prepare the numerical input snapshot.
7. Normalize constraints, trace callables where possible and precompile them.
8. Store programs, lookup indices and metadata in Simulation.

The source index defines the initial medium. If absent, ambient is the fallback.
An element with no declared index inherits the current medium. An Interface
checks its incoming index and changes the medium to its outgoing index.
Variable continuity expressions become equality constraints. A fixed mismatch
fails during build.

## Understand theta

`theta` contains independent variable Parameters. A shared Parameter has one
entry, even when several expressions use it. Fixed Parameters become constants.
Dependent expressions are calculated from theta; they do not get separate entries.

Indices follow first occurrence in the optical roots, then additional variable
dependencies from generated continuity constraints. `parameter_data` describes
those entries in theta order, including names, initial values and bounds.

User constraints cannot introduce variable Parameters outside this model.
Their programs may use a different local input order; Solver must align that
order before evaluation or fusion.

## Handle constraints

Use `system.require(...)` or `element.require(...)` to attach constraints before
build. Collection preserves composite-owned constraints and includes generated
Space constraints. Each optical object is collected once by identity.

Direct graph constraints are captured in a program. Unresolved Symbols fail.
A satisfied constant constraint is omitted. An unsatisfiable or non-finite
constant constraint fails build. A variable constraint may start unsatisfied.

Build traces constraint callables with a result tracer. Traceable expressions
are precompiled; callables with symbolic Python control flow remain deferred.
Their captured Python state remains the user's responsibility. Positions preserve ordering between
graph constraints and deferred callables. Add later optimization constraints
through `Solver.require()`; Simulation has no `require()` method.

## Run with parameter values

Each access to `simulation.initial_values` returns an editable copy:

```python
guess = simulation.initial_values
guess[lens.f] = 0.08
result = simulation.run(theta=guess.values)
```

Only Parameters in theta, or handles directly containing them, can be assigned.
Assigning to an expression is not supported. Guess edits do not change the
Simulation defaults or declarative Parameters. `simulation.run()` uses the
build-time defaults.

Run evaluates the optics program once. It reconstructs the numerical source,
then propagates it through each stage using that stage's matrix, length and
refractive index. Numerical values retain their backend types for differentiation.

The result records one run. Use another theta and run again to obtain a new result.

## Inspect the result

```python
result.source                 # same state as result.initial
result.final                  # final optical state
result.states                 # input, then one state after each stage
result.z                      # matching longitudinal coordinates

result[lens].before           # state entering the lens
result[lens].after            # state leaving the lens
result[lens].z_start
result[lens].z_end

result[lens, 1].after          # second occurrence, if the lens is repeated
result.stage(lens, occurrence=1).after
result.stages[0]              # first stage in propagation order
```

Omitting `occurrence` for a repeated element raises an error. `result[element]`
looks up an element by identity; use `result.stages[index]` for numerical indexing.

For stage `i`, its incoming state is `states[i]` and its outgoing state is
`states[i + 1]`. Their positions are `z[i]` and `z[i + 1]`. A zero-length element
can change the state without changing z. These are boundary samples. General
interior probing needs partial propagation models and is deferred.

State properties remain on the numerical states:

```python
result[lens].after.w
[stage.after.w for stage in result.stages]  # stage outputs
[state.w for state in result.states]       # includes the input
```

## Inspect the resolved optical system

`result.element_data` describes every resolved element (type, label, composite
path, generated Spaces included), in propagation order. An element inside a
composite has the path `(composite label, *attribute names)`, for example
`("Relay 1", "front")`; an element added directly and a generated gap have
`path=None`. Element labels and paths describe the resolved hierarchy; they do not
reconstruct the original composite constructor arguments. The evaluated optical outputs and compiled stages are
internal.

`result.theta` holds the values used for this run. `result.parameter_data` describes
their indices; its `value` fields are build-time defaults, not updated run values.
`str(system)`, `str(simulation)` and `str(result)` provide readable summaries of
the declared design, compiled model and boundary trace respectively.

See [the graph reference](graph-reference.md) for Nodes, bindings and programs.


## Solve and inspect

Solver(simulation) or Solver(system) creates one optimization model. Add objectives
with target() and constraints with require() before the first solve. The model
then stays fixed. A value-matching objective, `target(value=..., target=..., scale=...)`,
uses residual (value - target)/scale, where value is a graph expression or a
callable of the result. scale defaults to abs(target) where the target is not zero
(a relative error) and to 1 where it is, so give it explicitly for a zero target;
an explicit scale must be nonzero. Targets and scales may be numerical arrays;
broadcasting follows the numerical backend. Total cost is the sum of weight times
the squared residual entries. target=0 is valid.

`simulation.symbolic` is the result as graph expressions, bound to that simulation:
`sim = simulation.symbolic`, then `sim[lens].after.w`, `sim.final.z0` or
`sim[b].z_start - sim[a].z_end` can be used directly as objective values and in
inequalities. `solver.require(sim[lens].after.w <= 1e-3, label="aperture")` adds a
constraint; `label` names a single comparison. Callables of the result also work
and are needed when the code branches on result values.

Solution exposes x, cost, success, message, iterations, function_evaluations,
feasible, objectives, constraints, violations, simulation, value(quantity) and
run(). iterations and function_evaluations are the backend's own counts, or None
when a backend does not report them (SciPy reports both; NLopt reports only
evaluations; IPOPT's iterations come from its progress callback). value() returns the solved
value of a variable Parameter (or its direct handle) and evaluates any other graph
expression, such as a derived gap, with the interpreter. Solved values replace the
model's variable Parameters, fixed Parameters keep their current value, and the
Simulation's bindings resolve Symbols. A variable Parameter outside the model, an
unbound Symbol or a runtime result Symbol raises KeyError;
read result quantities from run(). There is no Solution indexing or value_of().
solution.simulation carries solved defaults, current parameter bounds and the full
constraints, including ones removed only for numerical optimization. It can be
passed to a new Solver. success is backend termination status; feasible includes
model constraints and parameter bounds within the reporting tolerance.

## Plot a result

result.plot(ax=None, show_elements=True, show_axis=True) and
from diffractix.plotting import plot; plot(result, ...) are equivalent public
entry points. Both return (figure, axes) and do not call show() or save files.
Matplotlib is imported lazily. Plotting supports concrete Ray, RayBundle and
GaussianBeam results, including numerical subclasses. z is shown in metres and
transverse quantities in millimetres. Bundles show up to 41 rays.

Only known Space intervals are interpolated. Thick elements such as GRIN retain
actual boundary states and a boundary-only region; plotting does not claim their
interior propagation. General arbitrary-z probing is unsupported. The old at(),
after(), z_at() and z_after() APIs are removed; use indexed stage views.
