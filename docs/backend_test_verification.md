# Numerical-backend test verification — 2026-10-19

## Changes

Backend tests now live in `tests/unit/backend/`, split into configuration,
Autograd, JAX, and numerical parity modules. The multi-station example has an
integration-level parity test. Shared numerical cases execute in explicitly
selected child processes, without recursive pytest runs or backend switching.

The cases cover three parameter vectors, compiled aliases and runtime bindings,
complex arrays, all built-in state types, interfaces, slabs, thick lenses,
apertures, actual solver preparation, gradients, full Jacobians, objective and
weighted-constraint Hessians, and nested/fourth derivatives. JAX-specific cases
also exercise complete JIT callbacks and batching.

Required Autograd is never silently skipped. Optional JAX is detected without
importing it; its specific and parity tests skip only when unavailable. Concrete
`require()` failures are checked, and unresolved traced predicates are expected
to be unchecked, with no callback assertion mechanism.

The literal-construction contract stays strict: backend arrays are not silently
converted to Python constants. Static test targets now use explicit `float(...)`
at the declaration boundary. The unchanged real example receives no such hidden
adaptation. Existing decorator assertions were moved into the backend-specific
cases, not discarded. Existing graph tests explicitly recognize native JAX weak
float promotion and its ragged-array exception type, while retaining numeric,
shape, ownership, and error-message checks.

## Results after the user's initialization fix

Project Python 3.14.7 already has Autograd and JAX installed. Pytest was loaded
from the existing ignored scratch dependency target; no virtual environment or
installed packages were changed. Separate processes ran the complete suite.

| Environment | Passed | Failed | Setup errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Autograd selected; JAX installed | 1,242 | 0 | 15 | 2 |
| JAX selected | 1,242 | 0 | 15 | 2 |
| `auto`, simulated JAX absence | 1,168 | 0 | 0 | 91 |

The dependency-absence run uses a scratch-only import hook inherited by child
processes. It demonstrates optional-dependency handling, not execution in an
actual Pyodide interpreter or a separately installed JAX-free environment.
Its skips comprise 89 optional JAX/parity cases and two optional optimizer tests.
The 11 explicitly selected Autograd unit tests also pass independently.

The complete runs took 92.68 s with Autograd selected and 164.49 s with JAX
selected. Both include explicitly selected child-process verification of both
installed backends. All 15 setup errors share the single example issue below;
the representative parity, differentiation, JIT, batching, configuration, and
validation-contract tests pass.

## Resolved issue and remaining example blocker

1. The user corrected the indentation of `_load_jax()`'s `return _Backend(...)`.
   Initialization, lazy/frozen configuration, and concurrent first use now pass.
   No library changes were made as part of the verification rerun.

2. The unchanged real multi-station example passes
   `target_focus.z_r` directly as an objective scale. Under JAX that is a backend
   scalar array, and strict `_make_literal()` correctly rejects it during
   symbolic tracing. Both complete reruns reproduce
   `TypeError: Expected numeric scalar, got <class 'jaxlib._jax.ArrayImpl'>`.
   The compatible call-site usage is `scale=float(target_focus.z_r)` at the
   example's static declaration boundary; the graph must not add implicit
   `.item()` conversion. Neither the example nor the library was changed to
   hide this failure.

Representative cross-backend/JIT verification passes. The **advanced example's
parity verification remains blocked**, so the full suite is not reported as green.
No failing checks have been marked xfail or skipped based on implementation
failure. After the example call site is addressed, rerun both complete backend modes;
the parity assertions report the operation, parameter-vector index, compared
values, and maximum absolute/relative discrepancy on failure.
