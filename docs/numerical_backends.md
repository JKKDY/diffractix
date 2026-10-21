# Numerical backends: integration and verification

Diffractix uses one lazy, process-wide numerical backend. `auto` selects JAX
when installed and otherwise Autograd. Importing Diffractix does not initialize
either backend. Select explicitly before numerical execution:

```python
import diffractix
from diffractix import backend

backend.configure("autograd")  # or "jax" or "auto"
```

Alternatively set `DIFFRACTIX_BACKEND`. Once a numerical operation or derivative
constructor initializes the backend, configuration is frozen. Autograd remains
a required dependency; JAX remains an optional extra. Packaging is unchanged.
The existing backend implementation was relocated from the accidentally named
`backend.py/` directory to the importable `backend/` package.

## Implementation boundaries

- Ordinary NumPy handles immutable snapshots, parameter metadata, bounds,
  concrete validation, constraint preprocessing, reporting, and optimizer adapters.
- `backend.xp` handles differentiable graph execution, beam properties, state
  stacking, objectives, and constraints. Solver derivatives use the existing
  backend wrappers. Static scaling coefficients stay on the host.
- Operation functions are resolved lazily, cached, and bound into compiled
  instructions. Constant folding and symbolic semantics are unchanged.
- Graph packing preserves strong NumPy dtype promotion and checks incompatible
  root shapes explicitly. Its scalar sequence fast path remains available.
- Concrete backend scalars can be captured as graph constants. Active tracers
  cannot be captured as declarative constants.
- Gaussian curvature branches use safe denominators and numerical selection;
  inactive divisions by zero no longer obstruct JAX tracing.
- Solver longitudinal-index lookup explicitly checks bounds, preserving errors
  despite JAX's different out-of-range array-indexing behavior.

There is no new simulator, PyTree registration, graph fusion, numerical cache,
automatic solver JIT, sensitivity strategy, or optimization adapter.

## Verification

Full suites ran in separate processes on Windows, Python 3.14.7:

| Numerical backend | Passed | Skipped | Failures |
| --- | ---: | ---: | ---: |
| Autograd | 1,155 | 2 | 0 |
| JAX | 1,155 | 2 | 0 |

The skips are the optional real IPOPT and NLopt integrations (packages absent),
not numerical-backend compatibility skips. Autograd also passes with warnings
treated as errors. One expected Autograd complex-adjoint casting warning is
locally filtered in the complex finite-difference regression, consistently with
existing complex differentiation tests.

Coverage includes runtime bindings, aliases, complex and array roots, dtype
promotion, all built-in states, composites, apertures, finite-curvature index
transitions, custom states/components, solver workflows, immutable snapshots,
weighted constraint Hessians, and nested/fourth-order derivatives.

New regressions verify isolated lazy/frozen configuration, simulated missing
JAX behavior without fallback for an explicit selection, complete callback JIT,
Python/custom states inside JIT, and analytical and central-difference derivative
checks. Existing differentiation tests now use the selected backend, rather
than accidentally differentiating JAX operations using Autograd.

At the identical advanced-example initial vector, maximum absolute differences
between Autograd and JAX were:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Propagated beam radii | 6.51e-19 |
| Objective | 7.64e-14 |
| Constraints | 1.74e-18 |
| Objective gradient | 1.73e-11 |
| Constraint Jacobian | 2.92e-16 |

All compared derivatives are finite. Complex optical derivatives also agree
with central differences at `rtol=2e-7, atol=1e-8`.

## JIT and custom-code contract

Complete numerical objective, gradient, constraints, Jacobian, objective Hessian,
weighted constraint Hessian, and nested derivative callbacks pass explicit JIT
tests. Python state/result objects can be created and consumed inside tracing;
the traced function must return numerical outputs, not a `SimulationResult`.

Automatic solver JIT is deliberately deferred:

1. Dynamic Python branches are supported by the existing eager interpreter
   fallback, but not automatically rewritten into JAX control flow. Their eager
   values and derivatives pass under both backends away from branch boundaries.
2. Concrete finite-value and optical-domain checks remain active during eager
   execution. Traced predicates cannot raise ordinary Python validation errors
   based on unknown values. Explicitly JIT-compiled numerical callbacks do not
   provide those runtime value checks; a future automatic JIT integration needs
   an eager validation boundary or an explicit checked-execution policy.
   Static shape checks and lookup validation remain available during tracing.
3. Complex expressions are supported, but a differentiated scalar objective must
   be real-valued. Select real/imaginary components explicitly when appropriate;
   do not silently discard an imaginary objective or claim holomorphic semantics.
4. JAX does not support NumPy object arrays or arbitrary-width Python integers.
   This difference is asserted explicitly, not skipped.

Custom numerical code should use `backend.xp`, avoid in-place array updates and
host conversion of active values, and keep traced numerical predicates out of
Python `if`. Ordinary parameter/element construction and static control flow
remain ordinary Python. Existing Autograd-only custom code is not rewritten.
See the official [JAX tracing documentation](https://docs.jax.dev/en/latest/tracing.html)
and [complex differentiation documentation](https://docs.jax.dev/en/latest/complex-differentiation.html).

## Advanced multi-station performance

Measurements use the unmodified example: 20 variables, 38 propagation steps,
15 objectives, and 25 retained general constraints. The exact existing physical
and scaled `Problem` objects are captured from `Solver.solve()`. Settings are
SLSQP, `maxiter=650`, `ftol=1e-11`, not the separate 500/1e-9 experiment.

The isolated harness is `scratch/solver_performance/backend_benchmark.py`.
Callbacks are warmed, then timed five times; tables report medians. JAX results
are synchronized with `block_until_ready()`. Forward timing includes constructing
the public result and extracting all beam radii. Full solve measurements are
single runs, not solve medians. BLAS/OpenMP thread counts are set to one for both
processes; XLA retains its own execution implementation. Optimizer trajectories
and backend threading must not be interpreted as matched-iteration speedups.

Versions: NumPy 2.5.4, SciPy 1.18.1, Autograd 1.9.1, JAX 0.11.2, with JAX's
64-bit mode enabled by the existing backend module. The final benchmark processes
ran sequentially, without overlapping test or benchmark workloads.

| Callback | Autograd eager (ms) | JAX eager (ms) | JAX warm JIT (ms) | JAX first JIT call (ms) |
| --- | ---: | ---: | ---: | ---: |
| Forward/result radii | 0.385 | 8.882 | 0.0124 | 375.8 |
| Objective | 0.543 | 9.255 | 0.0098 | 259.8 |
| Constraints | 0.509 | 9.782 | 0.0107 | 264.5 |
| Gradient | 6.817 | 145.447 | 0.1661 | 1,845.6 |
| Full constraint Jacobian | 46.344 | 80.914 | 0.2152 | 2,519.0 |

First JIT calls include compilation and execution. Warm timings exclude compilation.
The forward/objective/constraint/gradient/Jacobian programs in this table are
the physical callbacks, not an alternative numerical model.

| Solve mode | Solve time (s) | Iterations | Function evaluations | Final cost | Maximum violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autograd eager | 15.855 | 284 | 345 | 5.42e-13 | 0 |
| JAX eager | 89.948 | 341 | 430 | 9.57e-13 | 0 |
| Explicitly precompiled JAX | 0.609 | 301 | 382 | 2.68e-12 | 0 |

All three returned success. The violation calculation checks retained physical
constraints and parameter bounds at the unscaled solution. Optimized parameter
vectors differ; this is not a bit-identical optimizer-trajectory comparison.
The full physical vectors and initial numerical outputs are retained in the
scratch JSON records, and matching initial outputs were checked with
`rtol=2e-10, atol=2e-11`.

Problem preparation took 0.068 s for Autograd and 1.487 s for JAX. Explicit JIT
preparation of the four actual **scaled solver callbacks** took an additional
4.978 s. Their compilation is separate from the physical-callback diagnostic
compilation above; neither is hidden inside the reported warm solve time.
Preparation + scaled-callback compilation + solve is approximately 7.07 s for
the explicit JIT experiment. No automatic production JIT behavior is implied.

Conclusion: eager compatibility is established, but eager JAX is slower for this
Python-heavy workload. Explicit whole-callback JIT is promising. Until the
validation/dynamic-control-flow policy is implemented, select Autograd explicitly
when existing eager solver performance is the priority. This integration does
not change the requested `auto` preference for installed JAX.

## Test environment

Existing virtual-environment dependencies were not modified. The project Python
already contained both numerical backends; pytest was installed only into the
ignored `scratch/test_dependencies` target and used by the scratch test runner.
The final complete runs reported:

```text
DIFFRACTIX_BACKEND=autograd: 1155 passed, 2 skipped (warnings treated as errors)
DIFFRACTIX_BACKEND=jax:      1155 passed, 2 skipped (no warnings)
```
