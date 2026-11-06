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

This is a reference for the current rewrite. Package imports and the
Simulation-to-result handoff still need integration checks. The examples below
describe the API being connected. Solver integration is the next step.

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
position; build inserts a Space for the gap. Inputs, elements and composites
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
7. Normalize constraints and precompile direct graph expressions.
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

Context callables remain deferred for Solver tracing. Their captured Python
state remains the user's responsibility. Positions preserve ordering between
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

The result also carries the evaluated optical outputs and their metadata:

```python
i = 0
element = result.element_data[i]
step = result.optical_stages[i]
values = result.optical_values

parameters = {
    name: values[index]
    for name, index in zip(element.parameter_names, element.parameter_indices)
}
matrix = tuple(tuple(values[index] for index in row) for row in step.matrix_indices)
length = values[step.length_index]
outgoing_n = values[step.refractive_index_index]
```

`ElementData.parameter_indices` address optical outputs, not theta. They include
fixed values and dependent expressions. Empty optional fields are omitted.
Element labels and paths describe the resolved hierarchy, including generated
Spaces; they do not reconstruct the original composite constructor arguments.

`result.theta` holds the values used for this run. `result.parameter_data` describes
their indices; its `value` fields are build-time defaults, not updated run values.
`str(system)`, `str(simulation)` and `str(result)` provide readable summaries of
the declared design, compiled model and boundary trace respectively.

See [the graph reference](graph-reference.md) for Nodes, bindings and programs.
