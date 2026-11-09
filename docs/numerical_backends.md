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
Numerical differentiation and JIT are selected through the backend/ package.

## Implementation boundaries

- Ordinary NumPy handles immutable snapshots, parameter metadata, bounds,
  concrete validation, constraint preprocessing, reporting, and optimizer adapters.
- `backend.xp` handles differentiable graph execution, beam properties, state
  stacking, objectives, and constraints. Solver derivatives use the existing
  backend wrappers. Static scaling coefficients stay on the host.
- Operation functions are resolved lazily, cached, and bound into compiled
  instructions. Constant folding and symbolic semantics are unchanged.
- Program evaluation returns native root values as a tuple, without common-dtype
  packing. Optical propagation consumes these values individually.
- Concrete backend scalars can be captured as graph constants. Active tracers
  cannot be captured as declarative constants.
- Gaussian curvature branches use safe denominators and numerical selection;
  inactive divisions by zero no longer obstruct JAX tracing.
- Result stage/occurrence lookup checks indices on the host, preserving errors
  despite JAX's different out-of-range array-indexing behavior.

Solver caches complete numerical callbacks for the selected execution mode.
Changes to initial values or parameter bounds reuse that numerical core;
constraint reduction remains outside JIT. Switching between eager and JIT modes
rebuilds the cache. Omitting use_jit retains the previous mode, or uses eager on
the first solve. Eager and JIT caches are not retained side by side.
Graph programs are remapped and fused during preparation; there is no separate
simulator or PyTree registration.

## Current verification

Latest recorded full-suite results, 9 November 2026: Autograd 1338 passed,
4 skipped; JAX 1340 passed, 2 skipped; no failures or warnings. Explicit
fourth-order tests were removed as outside current release scope, not fixed in
the numerical library. See [QA status](qa-status.md) for remaining release work.

## JIT and custom-code contract

Complete numerical objective, gradient, constraints, Jacobian, objective Hessian,
weighted constraint Hessian, and nested derivative callbacks pass explicit JIT
tests. Python state/result objects can be created and consumed inside tracing;
the traced function must return numerical outputs, not a `SimulationResult`.

Solver JIT is opt-in through `solve(use_jit=True)`; Autograd treats JIT as an
identity operation. Objectives and requirements are fixed once callbacks have
been prepared. The following restrictions apply:

1. Dynamic Python branches are supported by the eager numerical callable
   fallback, but not automatically rewritten into JAX control flow. Their eager
   values and derivatives pass under both backends away from branch boundaries.
2. Concrete finite-value and optical-domain checks remain active during eager
   execution. Traced predicates cannot raise ordinary Python validation errors
   based on unknown values. Checks inside a JIT core do not inspect unknown traced
   values. Solver also
   checks the concrete returned objective cost and constraint vector outside
   its JIT cores. Custom functions still need concrete boundary validation.
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
