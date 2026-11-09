# Solver unit-test review results

8 November 2026. Specification: solver.md. Production, integration tests, shared
helpers, dependencies and pytest configuration were not changed.

## Execution

Entire owned directory, separate processes, existing virtual environment:

- Autograd: **392 passed, 1 skipped**, zero warnings, 2.34 seconds.
- JAX: **393 passed**, zero warnings, 12.10 seconds.
- The Autograd skip is solely the JAX executable no-retracing test. It executes
  and passes under JAX. No behavior failures were skipped or marked xfail.
- Availability: Autograd and JAX installed; cyipopt and nlopt absent.
- Native IPOPT/NLopt installations are not needed for adapter unit tests; local
  fake native APIs exercise their callback wiring. Native convergence is left to
  conditional integration tests. The import-laziness check uses one fresh child
  process, since in-process adapter tests deliberately import fake native APIs.

Commands: set DIFFRACTIX_BACKEND to autograd or jax, then run
`.venv/Scripts/python.exe -m pytest tests/unit/solver -q --tb=short -p no:cacheprovider`.

## Consolidated and deleted tests

- Deleted test_constraint_reduction.py: 1,024 lines targeting removed AST reducer
  entry points and ParameterInfo snapshots. Current Program reduction tests retain
  the corresponding duplicate/intersection, exact box boundary, fixed snapshot,
  canonical ordering, equality rank, affine implication, interval, optical-scale,
  signed-zero, vector/empty and nonmutation regressions. Ported the independent
  two-dimensional half-plane feasible-region sampling case to
  test_program_constraint_reduction.py. No removed reducer compatibility APIs added.
- Deleted test_context.py: tests for the unused SolverContext and old result
  at/after APIs. Current result Symbol compilation and resolution are exercised
  through preparation and callback tests. The production context module remains;
  removing its exports is a separate library cleanup decision.
- Deleted test_parameter_bounds.py: removed _parameter_bounds helper and obsolete
  metadata fixtures. Current canonical bound overrides and nonmutation checks live
  in test_cache.py and Solution tests.
- Consolidated duplicate initial-array/bound validation into test_cache.py,
  retaining before-compilation/reduction/dispatch and invalid-later-solve checks.
  test_solve_validation.py keeps model and feasibility-tolerance validation.
- Removed duplicate eager dynamic-branch, nonfinite-output and shape-change cases
  from test_callbacks.py; test_execution_regressions.py covers these more thoroughly
  including JIT, vectors, labeling, overflow, and differentiation of checked outputs.
  Retained independent analytical two-variable and Gaussian derivative checks.
- Replaced obsolete imports, fake Simulation construction and old public model-list
  access with current APIs. Missing symbols/external Parameters use fresh Solver
  instances instead of clearing a removed public list.
- Removed the identity-only assertion requiring Solver orchestration to copy
  options. The agreed contract forwards options; lazy/native adapter tests still
  require independent copies and caller nonmutation.
- Removed real SciPy optimization from unit preparation/reporting checks. Controlled
  adapter results now verify physical-coordinate unscaling, status and reporting;
  actual convergence remains integration work.

## Coverage by module

### Objective, constraint normalization and callable signatures

Added supported direct/explicit/callable forms, invalid evaluator entries,
zero-argument comparison/Node rules, wrong bounds, optional/variadic/keyword-only
signatures, bound methods and callable objects. Value/target/scale (including zero
target) and residual shorthand agree with explicit Objective values, gradients and
Hessians. require rejects invalid batches atomically. Existing strict nextafter,
array comparison and normalization regressions are retained.

### Preparation and Program reduction

Retained canonical theta remapping, binding aliases, runtime Symbol types,
compile-time tracing versus dynamic fallback, error propagation, constant
constraints and declaration-position tests. Retained extensive independent
mathematical reduction tests and added half-plane sampling against separately
written inequalities. No algorithm-order tests were introduced.

### Solver lifecycle/cache/callbacks

Added fixed-model target/require rejection, constructor type validation, repeated
initial guesses and changed bounds with identical numerical core objects, removed
constraints restored on relaxed bounds, and expected rebuilding when JIT mode is
changed. Omitting use_jit preserves the last selected mode.

Vector duplicate constraints verify reduced values, Jacobian row selection,
intersected bounds and weighted Hessian multiplier expansion analytically.
The JAX test warms all six core callbacks, then changes guesses/bounds and checks
that propagation tracing does not recur. Eager/JIT callback analytical checks,
complex Gaussian propagation, higher derivatives, nonfinite validation and lazy
propagation are retained.

### Problem and Scaling

Retained analytical variable/constraint transforms, two-variable nonlinear
Hessians/Jacobians and multiplier transforms, roundtrips and SolverResult metadata.
Added Jacobian shape validation, unsupported transformation types, finite large
bound widths without overflow, and avoiding Jacobian calls for empty constraints.
Read-only owned scaling metadata and fallback scales remain covered.

### Backends

Retained the three-backend adapter matrix, SciPy equality/inequality split and
subset Hessian multiplier expansion, IPOPT exact triangular Lagrangian Hessians,
NLopt signed/two-sided/equality residuals and optional gradient-buffer behavior,
options/status/diagnostics contracts. Extended lazy forwarding to options=None,
missing-dependency checks to both IPOPT and NLopt, and independently verified
that importing solver does not import either optional optimizer library.

### Solution

Retained full-model declaration-order reporting, vector residuals/violations,
parameter-bound tolerance boundaries, success versus feasibility, immutable x,
canonical value lookup, updated Simulation metadata, shared optical result and
repeated-solve isolation. Backend handoff test now uses controlled results rather
than native optimizer convergence.

## Findings and remaining gaps

No production bugs were confirmed by the final unit suite. Initial failures were
obsolete imports, fixtures, removed methods and an over-specific options-copy
assertion, all resolved by updating the tests to the agreed contract.

Remaining work is integration: real optical inverse-design scenarios, complete
examples, cross-backend native convergence and conditional real IPOPT/NLopt runs.
The entire repository suite was not run; unowned legacy integration tests still
need their own migration. context.py remains unused legacy production code.
Documentation describing persistent dual eager/JIT callback caches is historical;
the current agreed behavior rebuilds when the execution mode changes.

Scope limits: no promise of detecting mutation in deferred Python closure captures;
no new general AST evaluator on Solution; no unsupported mathematical identity
simplification or arbitrary tolerance assumptions added to reduction expectations.
