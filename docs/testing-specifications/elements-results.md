# Elements and shared base: test changes and results

## Final verification after authorized import repairs

The parent subsequently repaired library imports only, with user authorization.
Normal collection now succeeds. Both backends execute **134 passed, 1 failed**
across core and elements. The failure confirms the empty-handle alias validation
defect described below: `validate()` does not raise UnresolvedInputError.
The earlier blocked-run results below are retained as investigation history;
the final run establishes a runtime failure, not merely an inspected suspicion.

## Scope

Only these test files changed:

- `tests/unit/core/test_base.py`
- `tests/unit/elements/test_optical_element.py`
- `tests/unit/elements/test_elements.py`

Production code, dependencies, configuration and shared helpers were unchanged.
There are now 94 test functions across these files (84 before this pass).
Parametrized cases are additional; collection did not finish, so this is not an
executed-case count.

## Improvements

- Replaced deleted discovery-hook fixtures with explicit Node declarations.
- Migrated removed graph.utils/compile_ast APIs to public graph errors and
  compile_graph/Program. Matrix tests consume native tuple outputs, while
  vector-valued derivative tests explicitly stack the numerical results.
- Renamed the requirements noun to constraints while preserving require().
- Kept the independent optical matrices, curved and planar refraction
  gradient/Hessian checks, and the GRIN zero-gradient Taylor-limit tests.
- Added discovery across inheritance, explicit metadata, specialized/optional
  annotations and ClassVar exclusion, plus ordinary-field preservation.
- Added failed assignment atomicity, clearing/repopulating stable handles,
  prevalidation of unknown selection names, and bulk-versus-explicit selection
  of empty/derived fields.
- Added tuple-view and instance isolation for attached constraints, including
  live expression links after parameter reassignment.
- Added validation of dependencies in length/index, required empty matrix
  entries aliased with an optional index, valid shared DAGs and complex
  coefficients, and invalid array-valued matrix coefficients.
- Added Plane physics, independent per-instance aperture symbols/complex power,
  domain constraints following reassignment, and ABCD invalid-shape atomicity.

Three element tests duplicating generic constraint attachment/isolation moved
into the stronger shared-base test. Two narrower variability tests were replaced
by parametrized checks covering both variable() and fixed() and failure state.
No analytical physics regression was removed or weakened.

## Execution

Both requested focused runs were attempted with the existing environment:

```powershell
$env:DIFFRACTIX_BACKEND='autograd' # repeated with 'jax'
.venv/Scripts/python.exe -m pytest tests/unit/core tests/unit/elements -q --tb=short -p no:cacheprovider
```

| Backend | Passed | Failed test bodies | Skipped | Collection errors |
| --- | ---: | ---: | ---: | ---: |
| Autograd | 0 | 0 | 0 | 1 |
| JAX | 0 | 0 | 0 | 1 |

Both stop importing `system/__init__.py`, which imports the removed ElementInfo
from system.info. No test bodies executed; the results do not establish numerical
correctness under either backend. No import shim or skip was added.

`python -m compileall -q tests/unit/core tests/unit/elements` and the scoped
`git diff --check` passed.

## Suspected production defect awaiting runnable collection

`OpticalElement.validate()` excludes a root when it is the same object as an
empty optional refractive_index. That exclusion is applied to matrix and length
roots too. A required empty matrix dependency can therefore evade validation:

```python
empty = InputNode(None)
element = ValidationElement(
    matrix_value=((1.0, empty), (0.0, 1.0)),
    refractive_index_value=empty,
)
element.validate()
```

Expected: UnresolvedInputError, since matrix[0][1] is required.
Source inspection predicts: no error because both aliases are filtered out.
The unsuppressed regression is
`test_optional_index_does_not_hide_the_same_empty_handle_in_required_matrix`.
This is **inspection-confirmed control flow, not a runtime-confirmed failure**;
the package import blocker prevents executing it. Possible impact: malformed
custom elements pass this validation stage and fail later during compilation.

## Remaining gaps and policy questions

- Full numerical execution and backend parity must be rerun after production
  import repairs; this task intentionally makes no such repairs.
- Exact-zero validation through context-bound Symbols remains unclear because
  element.validate() has no bindings argument.
- ABCD's documented output-index semantics and System's continuity policy need
  agreement; no test in this module chooses that System-level policy.
- Generic Node-containing container annotations are discovered broadly, but
  array/complex Parameters are intentionally outside the supported field model.
