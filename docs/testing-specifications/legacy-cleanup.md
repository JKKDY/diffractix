# Legacy test cleanup

9 November 2026. Test files and helpers only; no library changes.

The baseline full suite had 39 backend fixture errors plus a scalar-literal test
failure caused by removed compile_ast/CompiledAST APIs. Backend representative
fixtures also used System(input=...), numerical sources, ctx.after and old graph
metadata. Those bulk fixtures duplicated current graph, input, solver and optical
integration coverage and prevented independent backend checks from executing.

Removed the unused tests/helpers/solver.py and tests/helpers/simulation.py fake
legacy Simulation builders, the backend conftest that supplied obsolete bulk
fixtures, and the old representative/advanced helper programs. No remaining
caller used the advanced case. The current integration parity group remains.

Removed duplicate optical/solver cases from the backend unit files. Preserved
backend-specific contracts in focused isolated checks: explicit selection,
float64 JAX execution, identity/static-argument JIT behavior, analytical complex
derivatives, forward/reverse Jacobians, polynomial fourth derivative, vmap and
concrete/traced require semantics. Existing active tracer and owned constant
checks remain. Updated scalar constant compilation to compile_graph and canonical
parameter_indices. Graph/optical snapshot and derivative coverage stays in its
current unit/integration homes. No tests were skipped or xfailed to hide failures.

Two known real/complex fourth-order Autograd stacking failures are genuine
retained regressions, not legacy API tests. They remain unmodified. No new test
or production failure was intentionally discarded. A temporary parity helper
initialization-order issue during cleanup was corrected before final verification.


## Final full-suite results

- Autograd: **1306 passed, 2 failed, 4 skipped**, no setup/collection errors.
- JAX: **1310 passed, 2 skipped**, no failures or setup/collection errors.
- Both runs reported no warnings. The two Autograd failures are the retained
  fourth-real/fourth-complex cases in graph/test_compile.py (Autograd add.at
  raises TypeError: first operand must be array). The real backend-specific and
  current integration tests execute; the former 39 legacy fixture errors are gone.
- Commands: existing .venv/Scripts/python.exe -m pytest tests -q --tb=short
  -p no:cacheprovider, separate DIFFRACTIX_BACKEND=autograd/jax processes.


## Derivative-scope decision

On 9 November the user limited release verification to practically relevant
numerical differentiation. Solver algorithms require gradients, Jacobians and
Hessians. Removed explicit fourth-order compiler cases for all real/complex/array
variants, the standalone sinc fourth derivative and backend polynomial fourth
derivative assertions. First/second derivatives, representative third/nested
transforms and sinc removable-limit Hessian tests remain. This is a test-scope
change, not a library fix or promise that fourth-order evaluation cannot work.
The earlier two known Autograd failures are no longer part of the release suite.

After the derivative-scope change, full-suite verification passes without warnings:
Autograd **1304 passed, 4 skipped**; JAX **1306 passed, 2 skipped**.
