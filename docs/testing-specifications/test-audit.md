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

## 1. Library bugs found (need approval before fixing)

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

## 4. Questions for the maintainer

- **Q1.** Where should a concrete invalid Gaussian `q` (`Im(q) <= 0` or `q` real)
  be rejected: at construction, at `numeric()`/build, or on property access?
- **Q2.** Should `Parameter` reject NaN at construction, as it does other invalid
  values?
- **Q3.** GRIN requires its explicit `n` to equal the incoming medium, so a GRIN rod
  in air needs explicit `Interface`s on both sides. Is that the intended user
  model? If so, it needs documenting.
- **Q4.** Is an array-valued `target=` (with an array `value`) in value matching
  part of the contract?
