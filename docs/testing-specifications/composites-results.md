# Composite unit-test results

## Final verification after authorized import repairs

The parent subsequently repaired library imports only, with user authorization.
Normal collection now succeeds: **49 passed** under Autograd and **49 passed**
under JAX, with no failures or skips. The blocked runs below describe the earlier
investigation state rather than the final verification result.

## Changes

Updated only `tests/unit/composites/test_base.py` and `test_composites.py`.
The directory now contains 43 test functions (previously 32); parametrization
adds analytical cases without repeating test bodies.

- Retained live-handle, child-order, built-in wiring, ambient-symbol,
  variability and nested-path regressions.
- Extended prefix testing to a nested composite.
- Replaced the low-value repr substring test with an empty-composite contract.
- Replaced the private-only discovery case with a broader private/class-only/
  metadata exclusion case, avoiding two tests for the same exclusion.
- Made the custom OpticalElement fixture use dataclass(eq=False), consistent
  with optical identity semantics.
- Added repeated-child occurrence versus binding-deduplication checks, symbolic
  nested lengths, local ordered constraint attachment, instance isolation,
  recursive Node-valued bindings, mapping independence, and duplicate-key
  failures for both matching and conflicting values.
- Added independent aggregate matrices for Slab in air and water, ThickLens
  with both curvature signs and planar surfaces, Telescope afocal behavior,
  and FourF imaging magnification. Expected matrices use physical-angle optics,
  not production element matrices as their reference.
- Added a shared focal-Parameter compilation case that checks a single theta
  entry, live declarative updates, and unchanged compiled variable membership.

## Execution

Commands used `.venv/Scripts/python.exe -m pytest tests/unit/composites -q
--tb=short -p no:cacheprovider`, in separate processes selected by
`DIFFRACTIX_BACKEND`.

| Backend | Passed | Behavior failures | Skipped | Collection errors |
| --- | ---: | ---: | ---: | ---: |
| Autograd | 0 | 0 | 0 | 2 |
| JAX | 0 | 0 | 0 | 2 |

Both test modules fail ordinary import because `system/__init__.py` imports
`ElementInfo`, which no longer exists in `system/info.py`. No composite test
body executes. This is an existing production integration blocker, not a
numerical or test-setup failure. No import aliases, shims or skips were added.
Both edited files pass Python AST syntax validation; runtime correctness remains
unverified until production imports are repaired.

## Remaining gaps and uncertainties

- Shared descendants across separate nested composites have unclear binding
  deduplication policy. Tests cover direct aliases only, where behavior is clear.
- Cyclic composite containment has no specified error behavior.
- Mutation of composition after construction has no refresh API; tests do not
  make a mutable-hierarchy promise.
- End-to-end System flattening, continuity and propagation remain in the System
  and integration suites rather than being duplicated here.

No additional composite-specific production defect was confirmed because test
collection is blocked. Library source, dependencies and configuration were not
changed.
