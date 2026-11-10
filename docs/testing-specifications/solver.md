# Solver package behavioral specification

Current behavioral contract for the Program-based Solver.
Read the current source plus graph-reference.md, system-simulation-result.md and
numerical_backends.md. Examples awaiting migration may describe removed APIs.
This specification distinguishes agreed contracts from observations and open policy.

## Scope and test rules

Own tests/unit/solver exclusively. Keep integration tests, shared helpers,
configuration and dependencies unchanged. Local helpers are preferred; a real
minimal System/Simulation is preferable to an obsolete fake Simulation. Do not
change production except a necessary import correction, reported explicitly.
Delete tests for deliberately removed APIs and consolidate genuine duplicates,
but port their meaningful mathematical assertions when the behavior survives.
Retain regression failures; no xfail or skip to hide bugs. Optional-library
availability skips and JAX-only transformation skips are legitimate. Shared
numerical tests import transformations through diffractix.backend. Native
optimizer end-to-end convergence tests belong in integration; adapter unit tests
may use small purpose-built fake native APIs to inspect callback wiring. Such substitutes test native adapter wiring without requiring optional packages;
they must not conceal library failures or replace real integration tests.

Run the entire owned directory in separate Autograd and JAX processes if installed.
Report collection errors, passed/failed/skipped counts, meaningful coverage
changes, minimal reproducers and remaining gaps. Do not install
packages or rewrite integration tests. Existing backend isolation helpers may be
used only when needed (e.g. JAX tracing tests); avoid gratuitous subprocesses.

## Objective and callable normalization (objective.py, utils.py)

Objective(evaluate, weight=1, label=None) represents residuals, not a separately
defined objective cost. Evaluate is Node or callable; weight must be finite and
nonnegative (zero is valid). Cost is weight * sum(square(residual)), summed across
objectives. Test scalar/vector and empty residuals with analytical expectations.
normalize_to_objective accepts Nodes, Objective, one-argument callables, and
zero-argument callables returning Nodes. callable_arity accepts an unambiguous
zero or one argument signature; reject signatures accepting both or neither.
Test bound methods, callable objects and keyword-required arguments where useful.
prepare_objectives traces once using SimulationResultTracer, precompiles roots
and preserves labels, weights and original positions. Bindings resolve aliases;
fixed values are captured by compilation. Only SymbolicControlFlowError triggers
fallback, not unrelated user exceptions. Deferred Python callables retain normal
closure semantics. Unresolved non-result Symbols fail; runtime result Symbols
remain program inputs. Current source retains array-returning trace results as
callables: record this observation rather than demand automatic array tracing.

## Constraint normalization/preparation (constraint.py)

Constraint(evaluate, lower_bound=-inf, upper_bound=inf, label=None) applies bounds
componentwise; equal bounds mean equality. Reject wrong evaluator type, NaN
bounds, reversed bounds. Comparisons normalize directly; strict LT/GT use
nextafter at scalar bounds or residual zero. Test operand reversal, two Node
operands and array literal comparisons independently.
normalize_to_constraint accepts Comparison, Constraint or a zero-argument callable
returning Comparison; raw Nodes and one-argument callables require explicit
Constraint wrappers. A zero-argument Constraint evaluator must return Node.
prepare_constraints normalizes, traces where possible and returns Program,
ConstraintDescriptors and positioned deferred constraints. Positions survive
interleaving and removed constants. With tracer=None, retain callable constraints.
Satisfied constant constraints disappear; violated/non-finite constants raise
clear errors. Symbol aliases resolve through bindings, only recognized result
Symbols may remain unresolved. Canonical theta membership is validated later by
System/Solver remapping, not by this helper. Preparation must not mutate models.

## Program constraint reduction

reduce_constraints(program, descriptors, parameter_lower=..., parameter_upper=...)
returns selected original output indices and effective descriptors, preserving
declaration metadata and full input Program. One descriptor per output; canonical
bounds shape is (n,). Reject invalid shapes/NaNs/reversed bounds/impossible infinite
endpoints and invalid descriptor intervals. Empty input is valid.
Prove duplicate interval intersections, affine box extrema (including exact
boundaries), supported interval implications, dependent equalities and affine
implications involving other constraints. Preserve nonlinear/unknown runtime
expressions conservatively. Never remove both mutually redundant constraints.
Contradictions must not become feasible. Partial side reduction may replace bounds
but must preserve labels/position and feasible region. Test fixed snapshots,
reordered canonical indices, infinite bounds, signed zero, vector/empty outputs,
small optical scales, nextafter boundaries, unsupported operations, cycles only
if valid Program APIs allow constructing them. Keep independent feasible-region
or analytical checks; avoid asserting a particular internal pass sequence.
Existing test_program_constraint_reduction.py has substantial relevant coverage;
old AST reducers in test_constraint_reduction.py no longer exist. Port unique
mathematical regressions rather than recreate the removed entry points.

## Problem and Scaling

Problem contains physical x0/bounds, six callbacks and scalar constraint bounds.
variable_count/constraint_count reflect their arrays. Length mismatches raise;
Scaling is constructed unless supplied. Do not demand unspecified comprehensive
standalone Problem validation; Solver validates per-solve inputs.
Scaling: x=c+s*y; c=x0; finite nonzero bound widths yield half-width s; fallback
abs(x0) or 1 for zero/unusable width. Constraint center is zero; scale is max over
abs(J_ij)*s_j per row; zero/nonfinite sensitivity falls back to 1. Metadata arrays
are owned/read-only. Empty constraints must not call a Jacobian unnecessarily.
Check scalar/vector transformations, infinite/equal bounds and overflow-sensitive
finite widths. Objective is not rescaled. Gradient becomes s*g, Hessian D H D,
constraint Jacobian diag(1/t) J D and weighted Hessian D H_g(x,lambda/t) D.
Unscale reverses these transformations, including multipliers. Verify analytical
two-variable nonlinear derivatives and roundtrip. SolverResult conversion changes
only x/scaling metadata and preserves raw diagnostics, cost/status/message.
Unsupported objects fail TypeError. No result.scale/unscale API exists.

## Native backend adapters and lazy imports

All entries accept problem, method=None, options=None; copy options and let native
libraries handle native options. Public solver dispatch uses SciPy by default.
Importing solver must not require IPOPT/NLopt. Missing selected packages raise
helpful ImportError; test unrelated installed packages do not affect dispatch.
SciPy DEFAULT_METHOD is SLSQP: import the constant rather than duplicate its value
in default-specific assertions. Supported SLSQP/trust-constr only. Forward objective,
gradient/bounds/options; equality/inequality constraints use separate objects;
trust-constr gets exact objective Hessian and constraint Hessian whose subset
multipliers expand correctly to full constraint rows. Empty constraints create no
native object. Preserve SolverResult raw diagnostics/status/physical metadata.
IPOPT rejects non-None method, passes native bounds and initial values, applies
provided options. Callback Jacobian is row-major flattened; Hessian is objective
factor * objective Hessian + weighted constraint Hessian, packed lower triangle.
Statuses 0/1 mean success; byte messages decode. Current IPOPT has exact Hessians;
do not reinstate obsolete limited-memory-default expectations.
NLopt uses configured/default algorithm, finite upper/lower inequalities with
correct signs and duplicated two-sided rows, and one equality row per equality.
In-place gradient writes occur only when requested. Verify constraint_tol, copied
options, supported setters/native params, unknown method/options, status mapping.
Real IPOPT/NLopt smoke tests are conditional integration, not required unit deps.

## Solver lifecycle, callbacks and orchestration

Solver accepts exactly one System or Simulation. Builds System once; readable
simulation cannot be replaced. target/require chain and normalize before storing.
target supports residual Node/callable/Objective and value=Node-or-callable,target=...,
scale (default abs(target) for a nonzero target, else 1),label forms; target=0 valid;
scale=0 and ambiguous combinations fail. require(*constraints, label=None) names a
single comparison (label with several constraints or a Constraint fails).
Simulation.symbolic returns the result tracer: its expressions are valid Node
objectives and Comparison constraints.
None means omitted in this current API. Objectives/constraints are fixed once
cache preparation completes, including when backend later fails. A new Solver is
required for another mathematical model. No graph fingerprints or invalidation.
Simulation constraints precede explicit solver constraints. Programs remap into
canonical theta and merge runtime Symbol maps before fusion. External variable
Parameters fail rather than extend theta. Use graph/result Symbols and real
Simulation to test alignment rather than legacy context.py.
Solve validates no variables, empty model, wrong entries, finite/nonnegative real
non-bool feasibility_tolerance and x0/lower/upper shape (n,), finite x0, no NaNs in
bounds, ordered bounds, feasible initial vector before backend dispatch.
Overrides are per solve, never mutate defaults. Options/method are forwarded.
Build six callbacks once for a fixed execution mode: cost, gradient, objective
Hessian, full flattened constraints, Jacobian, weighted constraint Hessian.
Compile all rows independent of current bounds; select rows outside JIT. Validate
vector row mapping, empty outputs and multiplier expansion, including duplicates.
Bounds changes must restore constraints previously redundant. Same mode reuses
original callback identities with different x0/bounds. Switching JIT mode rebuilds
the cache by agreed rollback; do NOT test dual eager/JIT cache persistence.
JIT is explicit for all six callbacks. Do not silently omit it for dynamic Python
branches; JAX may reject those branches. Deferred callables must return numeric
outputs; runtime Node returns fail. Finite checks must reject scalar/vector NaN/Inf
and cost overflow, while preserving grad/Hessian. Constraint shapes must remain
stable after initial preparation. No simulation run for pure Parameter programs
or empty constraints. One result view per callback evaluation serves all dynamic
entries. Runtime result-dependent expressions run simulation as needed.
Optimization receives scaled Problem and returns physical SolverResult/Solution.
Prefer captured backend Problems in unit tests; optical end-to-end solves later.

## Solution and SolverResult (solution.py)

SolverResult is the frozen native return container (x/success/cost/message,
iterations/function_evaluations/raw/scaling; the counts are None when a backend
does not report them). No solver/result.py exists. Public imports remain through
diffractix.solver. Solution receives full compiled programs/descriptors, not ASTs
or SolverContext. Public API: simulation,x,cost,success,message,feasible,
feasibility_tolerance,iterations,function_evaluations,objectives,constraints,
violations,value(quantity),run().
No targets/parameters/value_of/__getitem__ compatibility APIs are intended.
Solution x owns a read-only finite canonical vector; reject shape/nonfinite values
and invalid reporting tolerance. Original result.x mutation cannot change x.
Updated Simulation has solved defaults and CURRENT solve bounds, full constraints
including reduction-removed ones, and leaves original Simulation/Parameters intact.
Report original declaration order, residual shapes, labels/weights/individual cost,
bounds/margins/violations. Parameter bounds count for feasible and violations with
same reporting tolerance; success is independent. Nonfinite reporting fails clearly.
value(quantity) returns the solved value of a canonical Parameter or direct
InputNode, and evaluates any other graph expression with the interpreter: solved
values for the model's variable Parameters, current values for fixed Parameters,
Simulation bindings for Symbols. A variable Parameter outside the model, an
unbound Symbol (defaults ignored) or a runtime result Symbol raises KeyError;
non-graph arguments raise TypeError. run returns one reused optical result; result-dependent
reports share it; pure parameter reporting must not propagate. Repeated solves
cannot retroactively change earlier Solution reports or bounds.

## Limits

The unused context.py module remains exported but is outside the rewritten
pipeline. The active interfaces use SimulationResultTracer and result views.
Python callable captures and shallow binding-AST snapshots retain the limits in
../qa-status.md. Adapters copy options; Solver orchestration forwards them.
No new signature or domain restriction should be inferred from an implementation
detail. Record unclear policy rather than inventing an assertion.
