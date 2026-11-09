# Integration verification

## Plotting fix follow-up

The metadata mismatch below has now been fixed: sampling and rendering both use
SimulationResult.element_data. The plotting import test also uses the current
System(optical_input=...) constructor. All integration tests plus the plotting
unit test pass: Autograd **96 passed, 3 skipped**; JAX **97 passed, 2 skipped**,
without warnings. The historical full-suite results below precede this fix;
the unrelated backend-fixture and fourth-order issues were not changed or rerun.


8 November 2026. Three agents owned separate test files after the parent completed
[integration.md](integration.md). No library, dependencies or configuration changed.

## What was tested

- Real optical compilation and propagation: Ray, RayBundle, GaussianBeam, custom
  state/element, composites, material transitions, generated gaps and repeated
  elements. Analytical optics, complex/float32 behavior, source theta, snapshots,
  derivatives and JIT checked together.
- Real small SciPy inverse design: source angle plus lens focal length, Gaussian
  waist matching, coupled mechanical gaps, continuity, system/element constraints,
  repeated occurrence expressions, vector constraints and physical reporting.
- Sad paths and lifecycle: invalid model/lookup/bindings, infeasible constraints,
  optimizer iteration limits, Python branch fallback and explicit JAX rejection,
  failed preparation retry, relaxed bounds restoring constraints, prior Solution
  isolation and parameter-bound feasibility.
- Backend integration: scaled mixed equality/inequality problems under SLSQP and
  trust-constr; independent cross-backend numerical/derivative/solve parity;
  conditional real IPOPT/NLopt; public plotting and sampling.

Obsolete legacy APIs and private compiler/performance integration tests were
removed or replaced. Useful independent optical and regression checks were
preserved. No failed behavior was hidden with xfail or discretionary skip.

## Combined integration runs

| Backend | Passed | Failed | Skipped | Warnings |
| --- | ---: | ---: | ---: | ---: |
| Autograd | 90 | 5 | 3 | 0 |
| JAX | 91 | 5 | 2 | 0 |

Two skips are absent IPOPT/NLopt packages. Autograd additionally skips the
JAX-specific Python-branch tracer rejection test. Both integrations collect cleanly.

## Confirmed production blocker

All five retained failures are public plotting/sampling: result.element_info was
renamed to element_data, but plotting/sampling.py:63 still reads the old name;
plotting/matplotlib.py:93 also reads it. Ray, bundle, Gaussian, physical segment
sampling and empty-path plotting all stop there. No code fix was attempted.
Assertions beyond that error cannot yet establish plotting correctness.

Minimal reproducer and detailed expectations are in
[integration-backends-results.md](integration-backends-results.md).

## Remaining scope

Examples and large multi-station optimization are not verified. Real IPOPT/NLopt
convergence requires their packages. Interior-z probing and mutable binding-AST
snapshot policy remain outside the current API contract. Known fourth-order
Autograd stacking is deferred. Small-model convergence is not evidence that all
optical optimization landscapes converge.

Full repository results follow below. Existing backend unit fixtures still import
compile_ast in the legacy representative helper group; that group was left
unchanged, and the integration parity group is separate. An old backend scalar
literal test also imports compile_ast. The plotting import unit test still uses
System(input=...), so its failure precedes actual plotting. These pre-existing
unit migration issues are distinct from the confirmed current-API plot bug.

## Scenario reports

- [Optics](integration-optics-results.md)
- [Solver](integration-solver-results.md)
- [Backends and plotting](integration-backends-results.md)


## Full repository runs

| Backend | Passed | Failed | Errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Autograd | 1292 | 9 | 39 | 4 |
| JAX | 1296 | 7 | 39 | 2 |

No collection errors remain. The 39 errors are backend-unit fixture setup failures
from removed compile_ast imports, not new integration failures. Both runs include
five current-API plotting failures and two stale unit tests (backend scalar literal
compile_ast import and plotting System(input=...) call). Autograd adds the two
previously deferred real/complex fourth-order stacking failures. The extra Autograd
skips are the solver-unit no-retrace check and integration JAX branch check.
The separate pre-change Autograd unit baseline was 1202 passed, 4 failed,
39 errors, 1 skipped; the full Autograd result adds exactly this pass's integration
90 passed, 5 failed, 3 skipped. No library fix or failure suppression was made.

Commands used the existing .venv/Scripts/python.exe with DIFFRACTIX_BACKEND set
per process, pytest tests/integration or pytest tests, -q --tb=short
-p no:cacheprovider. Runs reported no warnings. Examples were not run.
