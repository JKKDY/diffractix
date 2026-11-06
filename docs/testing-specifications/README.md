# Unit-test behavioral specifications

These specifications were prepared before delegating test implementation. They
describe the source inspected on 6 November 2026, during the compiler migration.
They are test-design references, not a declaration that the migration is complete.

During implementation the user authorized library **import changes only**.
The parent agent made those changes centrally. Normal package import now works;
the baseline blockers below are historical. No numerical algorithms, validation
rules or class implementations were changed. Missing legacy Solver dependencies
are deferred to the methods that still use them, not replaced by compatibility
implementations. Final execution results are in `review-results.md`.

## Evidence and authority

Read the module specification together with `../graph-reference.md` and
`../system-simulation-result.md`. Those working references describe the current
rewrite more accurately than README examples and historical backend reports.
The README still uses `System(input=...)`; the current constructor uses
`optical_input=...`. Do not restore removed APIs to accommodate old tests.

Each specification separates documented contracts, observations from source,
and ambiguities. Do not turn an apparent bug into the expected behavior. Keep
clear regression failures; record uncertain policy rather than inventing it.

## Ownership

| Agent | Exclusive test ownership | Specification |
| --- | --- | --- |
| Graphs | `tests/unit/graph/` | `graphs.md` |
| Elements and shared base | `tests/unit/elements/`, `tests/unit/core/` | `elements.md` |
| Composites | `tests/unit/composites/` | `composites.md` |
| Inputs | `tests/unit/input/` | `inputs.md` |
| System | `tests/unit/system/` | `system.md` |
| Simulation and Result | `tests/unit/simulation/` | `simulation.md` |

Reports go in `docs/testing-specifications/<module>-results.md`. Do not edit
another agent's files, production files, dependencies, test configuration,
shared helpers, or integration tests. Small local fixtures are preferred.

## Test quality and execution

- Keep independent analytical physics and derivative checks. Do not derive an
  expected result by calling the same production function being tested.
- Compare graph identities with `is` or IDs. Node equality is symbolic.
- Update obsolete imports and fixtures to current APIs. Explain removal of tests
  for APIs deliberately removed, such as `clone_ast` and `Simulation.require`.
- Assert meaningful values, shapes, ordering, error types, and state transitions.
  Avoid exact formatting, global label-counter values, or a private register
  numbering scheme unless needed to expose an algorithm defect.
- Do not add xfails, skips, fake package modules, monkey-patched import aliases,
  or exception swallowing to conceal failures. Ordinary numerical-backend
  isolation is allowed using existing helpers, but shared tests should use
  `diffractix.backend` rather than import JAX or Autograd directly.
- Run owned tests with `.venv/Scripts/python.exe -m pytest <owned paths> -q
  --tb=short -p no:cacheprovider`, in separate processes with
  `DIFFRACTIX_BACKEND=autograd` and `DIFFRACTIX_BACKEND=jax` when available.
  Do not install packages. Record collection errors separately from test failures.

## Baseline and known blockers

The baseline seven-folder run under Autograd stopped with **15 collection
errors**. No behavior tests executed. Existing tests import removed
`compile_ast`, `CompiledAST`, `Opcode`, `graph.utils`, `ParameterInfo`, and
`ElementInfo`, and construct the old Simulation interface.

Production issues found by inspection, intentionally left unchanged:

1. `system/__init__.py` imports removed `ElementInfo`. This blocks ordinary
   top-level package import, including input tests.
2. `simulation/simulation.py` imports removed `SimulationTrace` and a nonexistent
   `.initial_values` module despite defining `InitialValues` locally. Its `run()`
   references `SimulationResult` without importing it.
3. `simulation/__init__.py` imports removed `SimulationSource`.
4. Solver code still imports legacy graph APIs, including `graph.utils.ast_equal`.
   System constraint normalization imports that package locally.
5. `graph/program_utils.py` uses a `Hashable` annotation without importing it;
   Python versions with eagerly evaluated annotations can fail import.
6. `remap_parameters()` assigns the boolean result of a membership comparison
   to `parameter_id` through a walrus expression. Test canonical remapping
   mathematically; do not assert this incorrect behavior.

These are not authorization to fix production. Tests can be written and syntax
checked even if collection remains blocked. Report the exact blocker and tests
that should exercise the later fix. Do not claim blocked tests passed.
