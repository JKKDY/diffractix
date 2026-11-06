# Inputs unit-test results

## Changes

Only `tests/unit/input/test_input.py` and `test_gaussian_beam.py` were changed.
The four Input test files now contain **62 test functions**, compared with 47
before this work. Parametrization produces additional cases; these are function
counts, not a claim about successfully collected or executed pytest cases.

Existing independent Ray/RayBundle matrix, shape, statistics, gradient,
Jacobian, Hessian and JIT tests were retained unchanged. Existing Gaussian
factory mathematics and nested derivative tests were also retained.

The obsolete `graph.utils` error import now uses public graph exports.
The old test expecting `numeric(None)` to behave as an empty mapping was reduced
to the documented omitted-bindings case. The current signature accepts a
mapping; it does not advertise explicit None. This removes an obsolete setup
assumption, not an exception-path assertion for a supported input.

New tests cover:

- ParaxialState's abstract propagation contract, including its numerical type.
- Recursive aliases and Node-valued bindings using an immutable mapping; current
  Parameter values remain live between conversions without mutating bindings.
- Missing alias and empty graph dependencies, followed by successful repair and
  retry without reconstructing the declarative input.
- Backend float32 scalar identity/dtype through numerical conversion, and
  analytical gradients/Hessians plus JIT through Symbol bindings and numeric().
- Per-instance default Symbols and attached constraints.
- Inherited propagation of custom Ray, RayBundle and GaussianBeam subclasses,
  preserving numerical class, properties, ordinary metadata and unwrapped
  assignments.
- Gaussian state properties from an independent complex-q example in glass.
- Free-space beam-envelope/curvature formulas at negative, zero and positive z.
- Waist overlap from the independent radius-mismatch formula, symmetry and unity.
- Overlap invariance through a common lossless lens/drift transformation.
- Concrete overlap medium/wavelength incompatibility and invalid inverse-q
  imaginary parts.
- Finite first/second beam-envelope derivatives at the plane-curvature waist.

## Execution

After the parent applied the user's separately authorized production import
repairs, both installed backends were selected in separate processes with:

```powershell
$env:DIFFRACTIX_BACKEND = "autograd" # then repeat with "jax"
.venv/Scripts/python.exe -m pytest tests/unit/input -q --tb=short -p no:cacheprovider
```

| Backend | Passed | Failed | Skipped | Warnings | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autograd | 130 | 0 | 0 | 0 | 1.00 s |
| JAX | 130 | 0 | 0 | 0 | 22.42 s |

Initial attempts before the import repair reported 4 collection errors per
backend, one per input test module. The first production error was
`system/__init__.py` importing removed `ElementInfo` from `system/info.py`.
Those historical collection failures were not hidden by skips or import shims.
The final runs above use ordinary package imports and no collection-error bypass.

All four test files pass AST parsing and Python syntax compilation. Scoped
`git diff --check` passes. This subagent modified no production files,
dependencies, configuration or shared test helpers; the parent separately
owned the authorized import corrections.

## Remaining gaps and uncertainties

- The tested conversion and numerical paths pass under both backends, including
  analytical gradient/Hessian and JIT checks. No new input implementation bug
  was exposed by these tests.
- Direct numerical methods on declarative InputNode-backed state fields are not
  treated as supported numerical operations; tests use numeric() or numerical_type().
- Invalid bundle factory counts/radii/focal distances and statistics of empty
  bundles lack a documented domain/error policy and were not specified by tests.
- Custom propagation implementations that hardcode the public declarative class
  remain the author's responsibility; inherited built-in propagation is covered.
- Graph field support remains scalar-only. Ordinary complex and array fields
  are intentionally not turned into Parameters. Numeric conversion retains
  ordinary array references; System, not numeric(), owns build snapshot isolation.
