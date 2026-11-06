# System: behavioral specification

## References and ownership

Read `system/system.py`, `info.py`, `symbols.py`, `errors.py`, exports, and
`../system-simulation-result.md`. Own `tests/unit/system/` only. Private build
passes can be tested directly to isolate responsibilities; use real graph and
metadata objects rather than recreating an obsolete Simulation with mocks.

## Construction and declarative model

```python
source = Ray(x=1e-3, theta=0.01, n=1.0)
lens = ThinLens(f=0.1)
model = System(Space(d=0.05), lens, optical_input=source)
model.add(Plane(label="Image"), z=0.2)
simulation = model.build()
```

Current API: System(*elements,optical_input=None,ambient_n=1.0), add(*elements,
z=None,index=None), add_input(*inputs), bind(key,value), set_ambient_n(value),
require(*constraints). Mutators return self. No old input= keyword,
add_input_beam, add_context, compile_context or requirements aliases.

add accepts elements/composites or nested iterables, preserving order and
composite identity. Strings/bytes and non-elements fail. Validate the whole
batch before mutating placements. Absolute z applies to one flattened addition
only; may be a Node or real finite nonnegative scalar, not bool. Scalar invalid
positions fail at add, not later validation. Current index is integer (not bool),
with range 0..len inclusive; invalid type/range fails atomically. Negative and
oversized indices differ from the historical Python-list-insertion proposal;
record the policy conflict instead of silently legitimizing either version.

add_input accepts one ParaxialState or one iterable containing it. Multiple
inputs, including separate repeated add_input calls, fail. No input fails build.
placements/elements expose tuple views. CallSite records source filename,
positive line and per-source-line count; compare filename to the caller's code
filename, not platform assumptions about cached bytecode paths.

Ambient n is a fixed Parameter by default; a supplied Parameter keeps identity
and variable status. Concrete ambient value must be finite positive real.
bind accepts hashable keys and Node or numeric Scalar except bool: real values
become named fixed Parameters, complex constants become Literals, Nodes retain
identity. Explicit rebinding replaces the System entry. bindings is a read-only
view. Object-contributed and System bindings are combined at build; duplicate
keys from distinct contributors raise, rather than choosing an override.

## Constraints and collection lifecycle

system_constraints are direct raw attachments; constraints collects them first,
then input constraints, then elements/composites in declared recursive order.
Collection of resolved placements additionally includes generated Spaces. Each
optical object contributes once by identity, including nested children. Composite
constraints must survive flattening. Never mutate original constraint lists.
Malformed constraints fail normalization during build; no backend is run.

`_compile_constraints(constraints,bindings,parameter_indices)` normalizes via
solver.constraint, splitting direct Nodes from deferred callable Constraints.
Store declaration positions for both to preserve interleaving and labels/bounds.
Zero-argument normalization follows normalize_to_constraint's contract; a raw
one-argument callable is not automatically a Constraint.

Direct graphs must resolve Symbols explicitly and may not introduce variable
Parameters outside canonical model theta. Fixed-only constraints are evaluated
at build: satisfied omitted, violated raises 'unsatisfiable constant constraint',
nonfinite raises clearly. Variable constraints may start unsatisfied. A deferred
callable retains ordinary user closure semantics; System does not trace it.
Compiled constraint programs currently have local theta mappings, which Solver
must align later. Do not assert they already accept full canonical theta.

## Resolution, compilation and metadata

_validate aggregates leaf validate() failures with location/path information;
it does not replace structural element validation. _resolve_elements flattens
composites in walk order, preserving identity, dotted paths and CallSite. Absolute
z belongs only to the first leaf; subsequent physical lengths determine layout.
_resolve_layout inserts Space(d=requested_z-current_z), then advances by each
element length. Generated constraints enforce nonnegative gaps, including
dependent gaps during optimization. Resolution does not mutate declared objects.

_resolve_refractive_indices starts from the source index; absent index falls back
to ambient. None element index inherits current medium. Interface requires n1
and outputs n2; explicit ordinary indices require continuity. Fixed mismatch
fails with location. Variable residuals become labeled equality Constraints,
even if initially mismatched. Inherited indices are baked as upstream expressions,
not assigned into declarative handles. Bindings resolve index Symbols.

_compile_optics outputs one native tuple Program: stage A,B,C,D,length,n;
source graph constructor fields; resolved element inspection fields; additional
variable dependencies from continuity. Canonical theta is deterministic first
occurrence across those roots. Shared Parameter identity gives one slot. Unused
bindings do not add variables. ParameterData includes variable theta entries
only, in index order, with build-time value and bounds. ElementData contains
identity, position, type,label,path, variable IDs, parameter names and output
indices (not theta indices). Empty optional handles are excluded. All inspection
fields are evaluated, including expressions not used in a matrix.

Sources are converted numerically then deep-copied once at build. Ordinary array
fields must be detached from the declarative input. Programs capture expressions
and fixed values. Optics cannot retain unresolved runtime Symbols at build.
build passes eight products to the new Simulation constructor. Build/run must
not mutate original parameter state, handles, constraints, bindings or arrays.

## Existing tests and priorities

Existing tests are largely tied to _compile, ParameterInfo, location_map and
legacy constructors. Rewrite around current passes, preserving layout algebra,
snapshot expectations and canonical indexing checks. Add external variable
rejection, mixed graph/callable order, constant infeasibility/nonfinite handling,
generated gaps, source-led medium, continuity-only theta dependencies, unused
binding exclusion, deep-copied arrays and repeat-build nonmutation.

## Open questions and known blockers

The shared README lists production import blockers. Do not patch them in tests.
No policy is established for a source with n=None that later reconstructs with
None even though medium resolution uses ambient. Independently canonical
constraint mapping and solver migration are outside this module. A shared leaf
appearing under multiple composites can duplicate contributed binding keys;
record this unresolved collection policy. Variable exact-zero singularities
and direct post-construction invalid Parameter mutation need separate audit.
