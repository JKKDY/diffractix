# Integration: numerical/native backends and plotting

8 November 2026. Test-only changes; no library, configuration or dependency edits.

## Tested

- Real SciPy SLSQP and trust-constr convergence, including scaled physical
  coordinates and analytical constrained quadratic optima. Mixed equality and
  inequality rows run through real native adapters with exact Hessian callbacks.
- Native iteration-limit termination reports an unsuccessful finite result.
- Real NLopt/IPOPT smoke tests remain conditional on their packages. Both are
  absent here, producing two legitimate skips in each numerical-backend run.
- Separate Autograd/JAX processes execute the same small Program, Gaussian free
  propagation and real source-position inverse-design solve. Values, gradients,
  Jacobians, Hessians and JIT values agree; primal graph/Jacobian, Gaussian widths
  and complex q, and solved ray position also have independent analytical checks.
- Complete Gaussian solver callbacks and nested derivatives, waist curvature
  factories, concrete invalid medium/wavelength/q validation, numerical custom
  states and RayBundle JIT propagation, and complex aperture/lens derivatives
  against central differences. Updated declarative-source/result-stage APIs.
- Intended plotting contracts: Ray, RayBundle and Gaussian results render into
  supplied axes without showing figures or mutating results; physical ray sampling
  across a lens; empty paths produce boundary samples without invented trajectories.
  These five checks currently expose one production issue described below.

## Consolidation

Removed test_backend_parity.py's advanced 20-variable example fixture and folded
parity into test_numerical_parity.py using current self-contained small cases.
The old numerical parity matrix depended on removed graph/System/result APIs;
replaced it with three coherent graph/optics/real-solve scenarios. Kept separate
numerical backend regressions, updated inputs and canonical theta indexing, and
made Invalid q explicitly expect ValueError rather than Exception.
Consolidated 200+ lines of obsolete plotting assumptions into three complete
state-family scenarios plus physical sampling and empty-path checks. No plotting
compatibility shim or fake result metadata is added.

Added an integration group to tests/helpers/backend_cases.py, leaving the
representative group's keys and behavior unchanged because unit/backend fixtures
also depend on it. That legacy group still imports compile_ast and other removed
APIs and caused 39 pre-existing unit fixture errors in the parent's baseline run.
Migration of that group is deferred separately; the new integration tests do not
execute it or expensive advanced examples.

## Runs

Owned files: test_backends.py, test_numerical_parity.py,
test_numerical_backend.py, test_plotting.py.
Command: existing .venv/Scripts/python.exe -m pytest <owned files> -q --tb=short
-p no:cacheprovider, separate DIFFRACTIX_BACKEND processes.

- Autograd: **15 passed, 5 failed, 2 skipped**, no warnings (7.01 s).
- JAX: **15 passed, 5 failed, 2 skipped**, no warnings (17.72 s).
- After tightening the Invalid q assertion, its focused Autograd rerun passed.
- diff --check clean for owned edits.

## Production issue retained as failing tests

Minimal reproducer:

```python
from diffractix import Ray, Space, System
result = System(Space(d=0.1), optical_input=Ray(x=0.001, theta=0.01)).build().run()
result.plot()
```

Expected: Matplotlib figure and axes with the physical ray trajectory.
Actual: AttributeError: SimulationResult has no attribute element_info.
Cause: plotting/sampling.py line 63 uses result.element_info after the rewrite
renamed metadata to result.element_data. plotting/matplotlib.py line 93 also uses
the removed name, so correcting sampling alone would not complete the fix.
All five plot tests fail at this first mismatch under both backends, including
empty systems. No xfail/skip was added; no production code changed.

## Limits

No native IPOPT/NLopt convergence verified without installed packages. No
large multi-station benchmark or example compatibility migration. Plot assertions
past the metadata mismatch cannot yet execute, so axes preservation and physical
sampling remain unverified until the library issue is fixed. No pixel snapshots,
performance thresholds or new numerical tolerances in production.
