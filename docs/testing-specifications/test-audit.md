# Test-suite audit

9 November 2026. Black-box audit: library source under `src/` was not read. The
baseline is commit `10ed9c7` plus uncommitted documentation edits. The test suite
was not rerun. One small probe script (a single Autograd process) and one JAX
process checked specific suspicions through the public API. Every library bug
listed below was reproduced; none has been fixed.

## Overall assessment

The suite is in good shape for a first release. Most numerical assertions use
independent analytical oracles rather than calling the code under test, and
derivatives are checked eagerly, under JIT and through nested transforms. Snapshot
and lifecycle contracts are explicit. The weak spots are a few untested public
components, operator coverage, and a few tests coupled to private implementation
order.

## Resolution status (9 November 2026, after commit `5827b78`)

The maintainer fixed B1–B3 and added regression tests for each. A black-box
re-probe confirms:
- B1: the build error now names the user's file and line.
- B2: `q=-1j` and `q=1.0` raise `ValueError: Invalid q-parameter` at construction.
- B3: `Parameter(nan)` raises, while `Parameter(inf)` is still allowed.

Settled answers to Q1–Q4:
- **Q1:** a concrete invalid `q` is rejected at construction (including
  numerical subclasses), in `w_R_from_q()`, and at build after a mutation.
- **Q2:** NaN is rejected at construction.
- **Q3:** GRIN in a different medium needs explicit `Interface`s on both sides.
- **Q4:** array `target=` is supported, with numerical broadcasting.

The re-probe found two new issues:

- **N1 (bug, needs approval): an array `scale=` is rejected with a raw NumPy
  error.** `solver.target(value=..., target=0.0, scale=np.array([1., 2.]))` raises
  `ValueError: The truth value of an array with more than one element is
  ambiguous` inside `target()`. The likely cause is the scale-is-zero check
  running on an array. The settled residual, `(value - target) / scale` with
  broadcasting, implies an array scale should work, checked elementwise for
  zero. Otherwise it should be rejected with a clear message. The maintainer
  needs to choose which.
- **N2 (cosmetic): wrong name in the NaN error.** For an unnamed parameter the
  new message reads "Parameter None value must not be NaN". Compile-time
  messages use `'<parameter>'` instead.

An array `target` works in a real solve (RayBundle focus recovers `f = 0.1`). An
incompatible target shape fails at solve time with NumPy's broadcast error.
**New coverage gap:** no test uses an array `target` or `scale`.

## 1. Library bugs found (fixed in `5827b78`)

### B1. Elements passed to the `System(...)` constructor report a library call site

```python
s = System(Space(d=0.1), ThinLens(f=0.0), optical_input=Ray(x=0, theta=0, n=1))
s.add(ThinLens(f=0.0))
s.build()
# 1. Placement #1 (ThinLens 'ThinLens1') added at ...\src\diffractix\system\system.py:56
# 2. Placement #2 (ThinLens 'ThinLens2') added at <user file>:4
```

`add()` records the user's location, but registration through the variadic
constructor records `system.py:56`. Every build error message (validation,
medium mismatch, generated gaps) then points into the library instead of the
user's design file. The existing test only exercises `add()`
(`unit/system/test_construction.py::test_registration_records_call_site_and_repeated_execution_count`).

### B2. A nonphysical Gaussian `q` fails inconsistently

| Source | Behaviour on build and run |
| --- | --- |
| `GaussianBeam(q=-1j, ...)` | No error. `initial.w` and `final.w` are silently `nan`. |
| `GaussianBeam(q=1.0+0j, ...)` | Raw Python `ZeroDivisionError: division by zero`. |
| `GaussianBeam.w_R_from_q(-1j, ...)` | `ValueError: Invalid q-parameter` (tested). |

The documented concrete-validation contract (`inputs.md`: "Concrete invalid q in
w_R_from_q raises") does not reach `.w` or construction. Expected: a consistent
`ValueError` for concrete invalid `q`. Traced values follow the existing
unchecked policy. Where to validate (at construction, `numeric()` or property
access) is a policy choice; see Q1.

### B3. `Parameter` accepts NaN at construction

`Parameter(np.nan)` and `Parameter(np.nan, lower_bound=0, upper_bound=1)` both
construct successfully. By contrast `Parameter(True)`, `Parameter(1j)`, `Parameter("x")`
and out-of-bounds values are rejected immediately. NaN is caught only later, at
compilation. NaN comparisons are always false, so the bounds check misses it.
Low severity, but inconsistent with the other construction checks.

Probes that confirmed correct behaviour (no bug): exact and negative-zero `f`/`R`
validation; variable `f` at zero; mutated unbounded `NaN`/bool/complex/str values at
compile; residual sign `(value - target) / scale` with a negative scale; Mirror
unfolded propagation; Telescope, Galilean Telescope and FourF matrices; a negative
Telescope spacing failing build; GRIN/ABCD explicit-`n` continuity errors; removed
`at`/`after`/`z_at`/`z_after` being absent; JAX defaulting to float64 without
`JAX_ENABLE_X64`.

## 2. Test-quality issues

| # | Where | Issue | Suggested action |
| --- | --- | --- | --- |
| T1 | `unit/solver/test_program_constraint_reduction.py`: `test_affine_forms_are_shared_by_the_box_equality_and_lp_passes`, `test_reduction_stops_when_an_earlier_pass_removes_every_output`, `test_equality_basis_eliminates_exact_dependencies_without_an_lp` | Patch private passes (`_program_affine_forms`, `_program_intervals`, `_reduce_equalities`, `linprog`) to assert internal pass order or call counts. `solver.md` says to avoid this. Any reducer refactor breaks them without a change in behaviour. | Delete them, or keep only their `indices`/`active` outcome assertions. |
| T2 | `integration/test_system_build.py::test_invalid_models_fail_before_propagation` | `pytest.raises((ValueError, KeyError))` with no `match`, so any unrelated error passes. | One expected type and message per case (missing input → `SystemValidationError`). |
| T2 | `integration/test_simulation.py::test_empty_path_and_invalid_theta_do_not_corrupt_next_run`; `test_system_build.py::test_nested_composite_paths...` (`len(...) >= 1`) | Weak oracles. | Add `match=`; assert the generated-gap descriptor explicitly. |
| T3 | `helpers/backend.py::run_python` | Always sets `JAX_ENABLE_X64=true`, so `unit/backend/test_jax.py`'s float64 assertion cannot detect the library failing to enable 64-bit itself. Verified correct today. | Don't set the variable in children, or add one child without it. |
| T4 | `unit/system/test_compilation.py` (`compile_optics` helper plus most tests) | Calls six private build passes in sequence, duplicating `build()`. A pass-signature refactor breaks many tests with no behaviour change. | Allowed as white-box, but give each user-visible rule (medium resolution, layout, snapshots) at least one `build()`/`run()` counterpart. Most already have one in integration. |
| T5 | `tests/README.md`; `helpers/backend.py` | README references the deleted `integration/test_backend_parity.py` and the "advanced example executed unchanged". `example_before_solve()` is now dead code that `exec`s examples. | Update the README; delete the helper. |
| T6 | `unit/solver/test_callbacks.py::test_full_callbacks_match_analytical_values_and_derivatives` | The `use_jit` parameter only wraps eager callbacks in an external `jit`; the Solver's JIT path is covered elsewhere. | Rename the parameter or the docstring. |
| T6 | `unit/simulation/test_simulation.py` | Fixtures use legacy `lambda ctx: ctx.theta[0]` constraints (never called) and assign to `Simulation` attributes after construction. | Use neutral callables; build a new `Simulation` instead of mutating one. |
| T7 | Several integration files (`test_system_build.py`, `test_source_compilation.py`, `test_sequence_propagation.py`, `test_element_propagation.py`) | Unspaced formatting (`a=b,c=d`), unlike the rest of the suite. | Cosmetic; run a formatter when convenient. |

## 3. Coverage gaps (highest value first)

1. **Graph operator table.** The `Op` enum has 21 operations. `tan`, `sinh`, `cosh`
   and `tanh` are never tested, and `cos` is tested only indirectly through GRIN.
   Most operations (`abs`, `max`/`min`, `log`, `sqrt`, `sigmoid`, `pow` with a
   variable exponent, `div`) have value checks but no derivative checks. Add one
   parametrized test: for each op, compare value, gradient and Hessian against a
   NumPy or analytic reference through the interpreter, `precompile` and
   `compile_graph`, eagerly and under JIT.
2. **Public components never built through a System:** `Mirror`, `Telescope` and
   `FourF` appear only in unit matrix tests. The probes show they work end to end,
   so these tests would be cheap: Mirror's unfolded `z` and angle, a Galilean
   telescope (`f1 < 0`), and negative total spacing failing build.
3. **B1 regression:** call sites for elements passed to the constructor.
4. **Settled continuity policy:** an explicit `n` on ABCD and GRIN must match the
   incoming medium; a variable `n` becomes an equality constraint. Unit tests
   parametrize only `Space` and `Interface`.
5. **Gaussian validity:** whatever B2 decides, cover construction, `numeric()`,
   `.w` and `.R` for `Im(q) <= 0`.
6. **Plotting:** only Ray sampling is checked numerically. Missing:
   - Gaussian envelope samples against `w(z)` in millimetres
   - `ax=None` creating a figure
   - `show_elements` / `show_axis`
   - the 41-ray bundle cap
7. **Mutated-Parameter validation:** value cases use only bounded parameters.
   Add an unbounded NaN case and bool/complex/str value cases (all behave
   correctly in the probes).
8. **Gaussian `overlap_with` tolerance boundaries:** just inside and just outside
   the `n` (1e-6) and wavelength (1e-9) tolerances.
9. **Low priority:** a guard that the removed `at`/`after`/`z_at`/`z_after` stay
   absent.

Correctly left untested, because policy is still open: binding deduplication
across composite branches, RayBundle factory domains and empty-bundle statistics,
and deep snapshots of binding ASTs.

## 4. Questions for the maintainer (answered; see Resolution status)

- **Q1.** Where should a concrete invalid Gaussian `q` (`Im(q) <= 0` or `q` real)
  be rejected: at construction, at `numeric()`/build, or on property access?
- **Q2.** Should `Parameter` reject NaN at construction, as it does other invalid
  values?
- **Q3.** GRIN requires its explicit `n` to equal the incoming medium, so a GRIN rod
  in air needs explicit `Interface`s on both sides. Is that the intended user
  model? If so, it needs documenting.
- **Q4.** Is an array-valued `target=` (with an array `value`) in value matching
  part of the contract?

## 5. Test work completed (9 November 2026)

Test and helper files only; `src/` was not touched. Only the affected files
were run:
- Autograd: 12 files, **405 passed**, no warnings.
- JAX: the new operator, ray, build and inverse-design tests, plus every
  helper-dependent backend and parity test: **93 passed**.

The full suite was not rerun.

**Cleanup**

| # | Change |
| --- | --- |
| T1 | Removed three reducer tests that patched private passes. A fourth called private `_reduce_equalities`; it duplicated the public `test_contradictory_dependent_equalities_raise`, so it was deleted. The pass-order test became `test_constraints_implied_by_parameter_bounds_are_all_removed`, which asserts only the outcome. |
| T2 | `test_invalid_models_fail_before_propagation` now expects one error type and message per case. The relay test checks the generated-gap value (0.05 − 2d) instead of `len >= 1`. `simulation.run([])` now matches its message. |
| T3 | `run_python` removes `JAX_ENABLE_X64` from the child environment, so float64 assertions now check the library's own setting. |
| T5 | `tests/README.md` updated; dead `example_before_solve()` removed. |
| T6 | `test_full_callbacks...` parameter renamed `wrap_in_jit`, with a docstring. Legacy `ctx.theta` lambdas in the Simulation fixtures replaced. |

Deliberately left alone:
- T4: private-pass coupling. The build-level counterparts exist in integration.
- T6b: the Simulation tests that assign attributes after construction.
- T7: formatting.

**Coverage added**

| Gap | New tests |
| --- | --- |
| 1 | `unit/graph/test_operations.py`: all 21 operations, checking value, gradient and Hessian against NumPy formulas. Each runs through the interpreter, `precompile` and `compile_graph`, eagerly and under JIT, with constant-operand variants. |
| 2 | `integration/test_ray.py`: Mirror unfolded propagation (three radii); Keplerian, Galilean and FourF traces; negative total spacing failing build. |
| 4 | `integration/test_system_build.py`: explicit ABCD, GRIN and Space `n` must match the incoming medium; a variable `n` becomes one continuity equality. |
| 6 | `integration/test_plotting.py`: Gaussian envelope samples against analytic `w(z)` and the plot's millimetre scale; `ax=None`; the `show_elements` and `show_axis` toggles; the 41-ray cap. |
| 7 | `unit/graph/test_compile.py`: NaN, bool, complex and str mutations on unbounded Parameters. |
| 8 | `unit/input/test_gaussian_beam.py`: `overlap_with` just inside and just outside its tolerances. |
| 9 | `unit/simulation/test_result.py`: guard that `at`/`after`/`z_at`/`z_after` stay absent. |
| Q4 | `integration/test_inverse_design.py`: an array `target=` solved per ray, with the residual and cost checked against the formula. |

Gaps 3 and 5 were covered by the maintainer's own regression tests in `5827b78`.
Array `scale=` is untested pending N1.

**New question for the library chat**

- **Q5.** `overlap_with` checks wavelength with an absolute tolerance of 1e-9 m
  (1 nm). That is about 0.1% at 1 µm, and beams 0.5 nm apart overlap silently.
  Is an absolute nanometre tolerance intended, or should it be relative? The
  new test encodes the currently documented values (`inputs.md`).

## 6. Final consolidation pass (9 November 2026)

I reviewed all 778 test functions by responsibility, after the maintainer's
N1, N2 and Q5 changes, which came with their own tests. The whole suite
collects 1,418 cases with no errors. Only the affected files were run, and
they pass: Autograd 683 (670 + the 13 other inverse-design cases), JAX 393.

**Redundancies removed or merged**

| File | Change |
| --- | --- |
| `unit/backend/test_helpers.py` | Deleted. It only tested the dead `example_before_solve` helper. |
| `unit/graph/test_construction.py` | `test_sigmoid` and `test_reverse_scalar_math` deleted; their values are now in `test_operations.py`. `test_unary_operations` became a structural check over all 14 unary builders (node type and operand identity). Section headers corrected. |
| `unit/elements/test_elements.py` | `test_abcd_matrix_override_rejects_wrong_shape` deleted; it was a subset of the atomic-rejection test. The three single-assert GRIN tests became `test_grin_declaration`. The Mirror, ThinLens and Plane "no constraint" tests became one parametrized test. |
| `unit/composites/test_composites.py` | Four `_structure` tests became one parametrized child-order test. Three `_can_be_variable` tests were deleted: generic base behaviour, already covered by `core`/`composites/test_base.py` and the wiring tests. The two spacing tests are covered by the reassignment and aggregate-matrix tests. The Slab and ThickLens ambient tests became one test parametrized over both. |
| `unit/input/test_gaussian_beam.py` | The old incompatible-overlap test and `w_R_from_q` rejection test are subsets of the newer tolerance and invalid-`q` tests. The combined tolerance test was reduced to the medium tolerance (symmetric); wavelength is covered by the relative-tolerance test. |
| `unit/solver/test_callbacks.py` | The empty-constraint test checks only derivative shapes. "No propagation for parameter-only models" stays only in `test_execution_regressions.py`, where it covers eager and JIT. |
| `integration/test_inverse_design.py` | My array-target test was removed; the maintainer's array target-and-scale test covers it. |

**Happy paths added**

- `test_quick_start_workflow_focuses_a_ray_onto_the_axis`: the README workflow on
  the current API, with default solver settings. It is precise to the README's
  printed 0.1 mm, and the declaration is left unchanged.
- `test_gaussian_beam_focused_by_thin_lens_follows_the_abcd_q_law`: a Gaussian
  beam through Space → ThinLens → Space in a System, compared at every boundary
  against the analytic `q` law.
- `test_converging_bundle_reaches_a_common_focus_through_the_system`: a factory
  bundle built and run end to end.

**Still deliberately not changed**

- T4: system unit tests use private build passes. Accepted white-box testing,
  with public counterparts in integration.
- T6b: Simulation tests assign attributes after construction.
- T7: formatting of some integration files.

None of these affect correctness or coverage.

**Full-suite verification:** the complete runs after this pass found no failures,
errors or warnings.

| Backend | Passed | Skipped | Time |
| --- | ---: | ---: | ---: |
| Autograd | 1,414 | 4 | 48 s |
| JAX | 1,416 | 2 | 105 s |

The skips are the absent IPOPT and NLopt packages, plus two JAX-only checks
under Autograd. This matches the earlier baseline.

## 7. Planned API changes and their test impact (design discussion, 9 November 2026)

These are maintainer decisions or leanings, recorded so the library chat and the
test suite stay in step. None is implemented yet, so no tests have changed for them.

| # | Decision | Test impact when it lands |
| --- | --- | --- |
| 1 | `solve()` defaults to JIT, and the keyword becomes `jit` instead of `use_jit`. `solver.py` imports the backend function `jit` at module level, so the import needs an alias (for example `backend_jit`) to avoid shadowing it inside the solve path. | Rename about 60 `use_jit` uses across 8 test files, 2 examples and `docs/numerical_backends.md`. The test that patches `solver_module.jit` (`test_execution_regressions`) must follow the new alias. 19 integration `solve()` calls pass no flag and will run JIT-compiled on JAX, so the JAX suite will get slower. The 9 tests that need eager execution already pass `use_jit=False`. |
| 1b | With a default of `True`, the "omit the flag to keep the previous mode" behaviour has to be redefined (today `None` keeps the last mode, and the first solve is eager). | Update `test_changing_jit_mode_rebuilds_cache_but_none_retains_previous_mode`. |
| 4 | A callable that cannot be JIT-compiled raises an error; `jit=False` is the explicit opt-out, with no separate strict flag. This already matches the current contract on JAX. The error should say to pass `jit=False`. | Tests that match JAX's raw "boolean conversion" message should match the new hint instead. |
| 4b | Add a `weight` option to the value-matching shorthand `target(value=...)`, and give more control over scaling. | Add tests for the weight (cost = weight · Σ residual²), including zero and invalid weights. |
| 2 | `Solution.value(expression)`: find the expression's Parameters and Symbols and evaluate it with the interpreter at the solution. Open details: a fixed Parameter's live value may differ from the value captured at build, and an expression that depends on runtime result Symbols needs the run. | Tests for derived lengths such as `850 mm − z6 − lens6.d`, for expressions with bound Symbols, and for the two open details. |
| 5 | Expose iteration and function-evaluation counts on `Solution`. | Test that they are reported and are positive for a real solve. |
| 5b | Solver locking: either allow `target()`/`require()` after a solve (invalidating the cache), or keep it fixed. `Solver(solution.simulation)` already covers reuse across simulations, so a per-simulation keyed cache is only needed if one Solver must swap simulations. | Update `test_model_definitions_freeze_after_preparation` if the lock is relaxed. |
| 3 | Thick-lens positioning (front vertex versus centre) is to be reviewed. | None yet. |
| – | GRIN rod without explicit Interfaces: low priority; a composite could package it. | None yet. |
