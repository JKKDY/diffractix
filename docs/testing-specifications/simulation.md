# Simulation and Result: behavioral specification

## References and ownership

Read `simulation/simulation.py`, `info.py`, `result.py`, exports and
`../system-simulation-result.md`. Own `tests/unit/simulation/` only; local new
test_initial_values.py/test_result.py files are allowed. Test the current
constructor with genuine Program and metadata fixtures, not the legacy graph,
source_spec, parameter_info or location_map interface.

## Constructor and lifecycle

Simulation receives eight keyword-only products from System:
element_data, parameter_data, callable_constraints (position,Constraint pairs),
constraint_descriptors, constraint_program, optics_program, optical_stages,
optical_inputs. Sequences are stored as tuples; programs retained directly.
There is no require method; attach future constraints to Solver. Deferred
callables retain ordinary closure semantics. Do not assert blanket deep
immutability of every object passed directly to this low-level constructor.

SimulationStep stores matrix output indices ((A,B),(C,D)), length_index and
refractive_index_index. SimulationInput stores a numerical initial_state and
field_indices (name,output-index), reconstructing with dataclasses.replace.
Updated graph fields use native numerical outputs; other constructor fields
are retained and init=False derived fields recompute through construction.

## InitialValues

```python
guess = simulation.initial_values
guess[lens.f] = 0.12
result = simulation.run(guess.values)
```

Each property access yields a fresh editable InitialValues with a copied float
vector in canonical ParameterData order. Internal defaults are read-only and
edits must not affect them or declarative Parameters. Indexing accepts a model
Parameter or InputNode directly containing it; membership is by identity, not
name/value/current variable flag. Missing/fixed-outside-model Parameters raise
KeyError; expressions, indirect handles and other keys raise TypeError.
Assignments require finite real scalar, not bool/complex/array/string; invalid
updates leave prior values unchanged. Bounds are intentionally not enforced
here, since a later solve can override bounds. `.values` is a numerical copy,
not another graph. Tests should include zero-variable initial vectors.

## Run and numerical invariants

run(theta=None) defaults to frozen build values. Exactly one optical input is
required. Evaluate optics_program exactly once, preserving native mixed
real/complex/array output types; reconstruct source, then propagate in stage
order. Each step reads its metadata indices rather than assuming a contiguous
layout. Pass outgoing n to state.propagate. Positions start at zero and accumulate
physical lengths, retaining duplicates for zero-length elements. Neither source
nor default theta is mutated. Repeated runs use supplied independent theta.

Return a SimulationResult with theta, parameter_data, element_data,
optical_stages, optical_values, z and states. A valid trace has N elements,
N stages, N+1 states and z positions. There is no generated per-state result
subclass or auto-stacked result.w property. Custom numerical ParaxialState
subclasses must work without Gaussian-specific code. Scalar functions extracting
numeric outputs from run should retain backend differentiation/JIT, including
real source wavelength alongside complex aperture coefficients. Do not require
JIT to return an arbitrary Python result object directly.

## Result inspection

SimulationResult snapshots metadata/value/state sequences as tuples, and makes
a backend theta array. Constructor rejects unequal z/states lengths, zero states,
states!=elements+1, stages!=elements, theta!=parameter_data. No extra numerical
shape validation is documented. Source and initial are states[0]; final states[-1].

```python
result[lens].before
result[lens].after
result[lens].z_start
result[lens].z_end
result[lens, 1].after
result.stage(lens, occurrence=1).after
result.stages[0].after
```

Stage i maps states[i]/states[i+1] and z[i]/z[i+1]. Element lookup is identity-based
through ElementData.element_id; never depend on live field values/equality.
Unknown element KeyError; repeated element without occurrence ValueError;
noninteger/bool occurrence TypeError; negative/out-of-range IndexError.
result[element] is not numerical indexing: use result.stages[index]. Malformed
tuple keys raise TypeError. at(element),after(element),z_at,z_after retain the
same boundary/occurrence semantics. General interior probing is deferred.

Element parameter_names zipped with parameter_indices retrieve evaluated
optical_values, including fixed and derived expressions. ParameterData.value
is the build-time default, not theta for the current result. Metadata alignment
must hold even for repeated elements and zero-length transitions.

## Display and testing priorities

str(simulation) summarizes compiled layout, values, variables and constraints;
str(result) uses result_columns on numerical states and labels/path/type for
locations. Check informative content and side-effect freedom, not table spaces
or global generated label numbers. Arrays must not break result formatting.
Plotting is outside ownership and remains coupled to the old result structure.

Replace obsolete constructor and removed Simulation.require tests while retaining
single evaluation, remapped-index propagation, snapshots, custom states,
occurrence/error, source and differentiation intent. Add editable-copy initial
values, tuple metadata isolation, Stage views, constructor length errors,
complex/native roots and derivative checks. Avoid depending on System.build for
all tests; direct minimal Programs make Simulation tests focused.

## Known defects and ambiguity

Production import blockers are in the shared README; leave them unchanged.
run references SimulationResult without an import. Add normal tests that would
expose this, rather than injecting that global in fixtures. at(numeric_z)
references nonexistent _probe; documentation defers probing, so report this
legacy path instead of inventing sampling semantics. SimulationInput.construct
uses shallow replace: ordinary mutable arrays may be shared between successive
run sources/results. Whether returned results may be mutated independently is
not fully specified; distinguish this from the documented guarantee that
mutating the original declarative input cannot affect built simulations.
