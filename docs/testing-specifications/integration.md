# Integration test specification

8 November 2026. Read solver.md, graph-reference.md and
system-simulation-result.md first. Current source is the rewritten API; historical
README/examples/helpers may be stale. Tests only, no production changes. Do not
change dependencies/configuration or unit tests. Delete/consolidate obsolete
integration tests when useful behavior is preserved; never hide legitimate bugs.
Use independent analytical/physical expectations, small deterministic optical
models, numerical tolerance appropriate to optimization, finite checks and
backend transformations from diffractix.backend. No benchmarks or large solves.

## Current contracts

System(*elements, optical_input=source), add/input/require/bindings as current
source. Node-annotated source fields are declarative. .numeric() is standalone;
System expects declarative OpticalInput, constructs numerical source per run.
Program is the compiled graph (precompile/compile_graph per actual exports),
parameter_data and optics_program.parameter_indices define canonical theta.
Simulation.initial_values is an editable COPY, .values is its numeric vector;
run(theta) returns SimulationResult. result[element].before/after/z_start/z_end,
result[element, occurrence] for repeats; initial/final/states/z remain available.
No legacy at/after, parameter_graph, steps/element_info, execution_context,
compile_ast, CompiledAST, OptimizationResult or Solution.targets/value_of aliases.
SolverResult and Solution are in solver/solution.py, public imports via solver.

Source index is initial medium (ambient when inherited/None); independent explicit
index mismatches at boundaries remain continuity constraints, variable initial
mismatches can be optimized; unsatisfiable fixed constraints fail at build.
System includes normalized/traced requirement Program; solver remaps/fuses its
own constraints. No external variable theta entries from ordinary constraints.
Traceable callables use result-view syntax and compile to Symbols; dynamic
callables fall back eagerly and must return numerical outputs, never Nodes.
JIT all numerical cores when requested, never silently bypass dynamic branches.
Same execution mode caches core callbacks across guesses/bounds; changing bounds
must restore necessary constraints. Changing execution mode rebuilds cache.
Solution x is physical/read-only; .simulation has solved defaults/current bounds,
full original constraints; .objectives/.constraints order by declaration;
.violations includes parameter bounds; success differs from feasible;
.value(parameter/direct handle) only; .run() returns solved optical result.
Mutable binding AST leakage TODO remains known, do not assert an invented snapshot
policy. Direct mutation of internal numerical snapshots is not a public guarantee.

## Workstream A: optics / compilation / propagation

Own test_system_build.py, test_simulation.py, test_source_compilation.py,
test_sequence_propagation.py, test_element_propagation.py, test_ray.py,
test_ray_bundle.py, test_input_propagation.py; all in tests/integration.
Cover actual System->Program->Simulation->Result: Ray, RayBundle, GaussianBeam,
custom dataclass optical state/element, nested composites, Slab/ThickLens and
repeated instances. Preserve analytical ABCD and physical angle/index conventions,
complex Gaussian/aperture cases, mixed real/complex output semantics, representative
float32 behavior, differentiation/JIT. Relevant boundaries: zero length/GRIN zero,
planar interfaces, inherited indices, generated gaps with dependent positions,
shared theta, source variables, ordinary mutable source-array snapshots.
Bad paths: invalid layouts/types/unresolved Symbols, ambiguous occurrence/missing
elements, fixed infeasible constraints and noncanonical requirement variables.
Consolidate old one-assertion cases into coherent end-to-end scenarios; avoid unit
API replication. Probe any discovered bugs with clear failing reproducer, no fix.

## Workstream B: real inverse design / lifecycle

Own test_inverse_design.py, test_solver_solution.py, test_solver_compilation.py,
test_solver_expression_evaluation.py, test_solver_jit.py,
test_snapshot_lifecycle.py, test_symbol_resolution.py; tests/integration.
Use small real SciPy solves (default plus a representative trust-constr case),
source and optical element parameter optimization, actual physical observables.
Verify targets + system/composite/element constraints, repeated occurrences inside
solver expressions, continuity enforced through optimization, coupled gaps,
vector constraints/partial reduction, scaled physical coordinates and Solution.
Include source theta and distance/focal length where analytical optima possible;
a small Gaussian focusing match suffices, no advanced 20-variable solves.
Verify fixed model lifecycle, bad input before native dispatch, failed backend or
infeasible termination separate from feasibility (controlled backend only if
needed), repeated solves with bound relaxation, immutable previous solutions,
parameter/source snapshots, bound reporting. Traceable vs numerical dynamic
fallback and explicit JIT failures; no Node-return fallback allowed. Runtime Symbol
and compile bindings must compose with actual optical run and reporting.
Prefer current backend run over nested subprocess parametrization. A JAX-specific
check may skip if unavailable/current backend is Autograd, with clear reason.

## Workstream C: native backends / parity / plotting

Own test_backends.py, test_backend_parity.py, test_numerical_parity.py,
test_numerical_backend.py, test_plotting.py; tests/integration.
May update tests/helpers/backend_cases.py only, exclusively owned here, to migrate
obsolete cross-backend cases. Do not modify other shared helpers.
Exercise small real SciPy solves and IPOPT/NLopt only if genuinely installed
(importlib.util.find_spec); optional packages missing means conditional skip.
Backend parity should use self-contained small models and independent analytical
values/derivatives, not stale examples or print parsing. Use existing process
isolation helpers only where backend comparison requires separate interpreter.
Check representative grad/Jacobian/Hessian and JIT equivalence, complex propagation,
real optical constrained optimization, reasonable optima/feasibility tolerances.
Plotting is currently to be inspected: if public plotting implementation was
removed, delete obsolete plotting tests and document this as deferred coverage;
do not recreate production plotting. If present test Agg figures/axes/data and
no mutation with actual results; avoid pixel snapshots.

## Reports and verification

Each agent owns docs/testing-specifications/integration-<optics|solver|backends>-results.md.
Record scenarios, tests removed/consolidated, runs passed/failed/skipped, minimal
bug reproducers with expected/actual behavior, policy ambiguity and uncovered areas.
Run owned files under Autograd and JAX separately using existing .venv/Scripts/python.exe
-m pytest ... -q --tb=short -p no:cacheprovider. No package installation.
Parent runs combined integration suite and unit+integration verification afterward;
full-repository pre-existing failures must be reported separately. Keep optional
backend absence skips; no new xfail/unjustified skip to silence failures.
