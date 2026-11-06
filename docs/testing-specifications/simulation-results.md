# Simulation unit-test results

## Changes

Replaced the old single Simulation test module with three focused modules:

- `tests/unit/simulation/test_simulation.py`: compiled execution and numerical
  source reconstruction.
- `tests/unit/simulation/test_initial_values.py`: editable canonical guesses.
- `tests/unit/simulation/test_result.py`: result validation and boundary views.

The previous suite had 37 test functions, built around deleted graph/source,
ParameterInfo, ElementInfo and location_map interfaces. The updated suite has
29 functions and **79 collected parametrized cases** using real Programs and
the current eight constructor products.

Removed tests of deliberately removed Simulation.require(), legacy context
mapping fields and generated result-type assumptions. Preserved their useful
independent intent where applicable: immutable build defaults, source/state
nonmutation, metadata alignment, numerical execution and occurrence lookup.

## Coverage

- Constructor tuple snapshots and retention of compiled programs.
- Zero-variable and zero-stage execution, invalid optical-input count, single
  program evaluation per run and deliberately noncontiguous stage indices.
- Sequential custom-state propagation with independent analytical outputs,
  passed outgoing medium, duplicate zero-length positions, independent runs,
  caller theta preservation and frozen default values.
- Numerical source reconstruction, init=False recomputation and numerical
  subclass preservation.
- Editable InitialValues copying, reordered mappings, direct handles, identity
  versus equal metadata, invalid keys, invalid scalar assignments with state
  preservation, changed live variability and intentional allowance for future
  bound overrides.
- Mixed real wavelength and complex aperture roots, including independent
  analytical gradient and Hessian expectations through Gaussian propagation.
- float32/float64 ray-bundle roots without unwanted scalar dtype promotion.
- Shared source/propagation theta: analytical values, gradient, Jacobian,
  Hessian, nested gradient and eager/JIT agreement through backend wrappers.
- Stage views, subscripts, repeated-element occurrences, legacy boundary
  accessors, identity lookup errors, integer/type/range errors and malformed
  tuple keys.
- Every documented result length invariant; tuple sequence ownership and an
  owned result theta copy without confusing current theta with metadata defaults.
- Scalar/array result display, valid state result_columns, frozen layout
  summaries and reconstruction of mixed graph/callable constraint order.

## Execution

After the parent repaired production imports only, normal collection and
execution succeeded. No import shims, collection-error continuation, skips or
xfails were needed.

```powershell
$env:DIFFRACTIX_BACKEND='autograd' # repeated with 'jax'
.venv/Scripts/python.exe -m pytest tests/unit/simulation -q --tb=short -p no:cacheprovider
```

| Backend | Passed | Failed | Skipped | Collection errors | Warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autograd | 79 | 0 | 0 | 0 | 0 |
| JAX | 79 | 0 | 0 | 0 | 0 |

The final focused runs took approximately 1.3 s and 6.3 s respectively.
AST syntax validation and scoped `git diff --check` also passed.

The Simulation agent changed only the three owned test files and this report.
All production import repairs were performed separately by the parent.

## Remaining limitations and coverage gaps

- Generic interior probing remains deferred. The legacy `result.at(numeric_z)`
  branch still references `_probe`, which the new constructor does not provide;
  its current failure is an AttributeError rather than a clear unsupported
  operation error. No sampling semantics were invented for it.
- Simulation.run still has a stale deferred SimulationTrace return annotation;
  ordinary execution works, but annotation introspection can fail until the
  annotation is updated separately.
- SimulationInput reconstruction uses shallow dataclass replacement. Arrays
  owned by the built source can be shared by successive reconstructed sources.
  Independent mutation of returned result arrays is not an established API
  guarantee. Tests preserve the narrower documented build snapshot contract.
- Plotting integration is outside this unit-test ownership and still needs the
  current Result metadata interface.
- System-to-Simulation build isolation is covered by the System agent; no
  end-to-end Solver migration or backend optimization was added here.

No numerical production defect was exposed by these 79 cases under either
backend. The legacy probing and return-annotation issues are findings from
inspection, not failures hidden from the test run.
