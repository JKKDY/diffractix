# Unit-test review: final results

## Work completed

All six behavioral specifications were written before any subagent was started.
Six module agents then worked with separate test ownership, in batches limited
by the available agent slots. The parent reviewed the changes, investigated
failure attribution, and ran the combined suites. No dependency, shared helper,
test configuration, example or integration-test file was changed.

During the task, the user authorized **library import changes only**. Seven
source files received import repairs. Comparing their Python ASTs with imports
removed confirms there are no non-import code changes. No validation,
optimization, propagation or solver behavior was repaired to make tests pass.
No new skips, xfails or compatibility shims were added.

## Final focused execution

Both backends are installed in the existing `.venv`; Python is 3.14.7.
Each backend ran in a fresh process. These are normal complete runs, without
collection-error continuation or partial-import workarounds:

```powershell
$env:DIFFRACTIX_BACKEND = 'autograd' # repeat with 'jax'
.venv/Scripts/python.exe -m pytest tests/unit/graph tests/unit/core tests/unit/elements tests/unit/composites tests/unit/input tests/unit/system tests/unit/simulation -q --tb=short -p no:cacheprovider
```

| Module | Autograd passed | Autograd failed | JAX passed | JAX failed |
| --- | ---: | ---: | ---: | ---: |
| Graphs | 200 | 31 | 202 | 29 |
| Elements and OpticalBase | 134 | 1 | 134 | 1 |
| Composites | 49 | 0 | 49 | 0 |
| Inputs and ParaxialState | 130 | 0 | 130 | 0 |
| System | 115 | 6 | 115 | 6 |
| Simulation, InitialValues and Result | 79 | 0 | 79 | 0 |
| **Total** | **707** | **38** | **709** | **36** |

There are **745 cases per backend**, zero skips, zero collection errors and no
pytest warnings in these focused runs. Final elapsed times were 6.29 seconds
for Autograd and 50.07 seconds for JAX. These timings are not benchmarks.

All changed and new Python files pass syntax parsing. The CRLF-aware whitespace
check passes (`git -c core.whitespace=cr-at-eol diff --check`); the default Git
whitespace check treats existing source CRLF endings on edited lines as trailing
whitespace. No Git configuration was changed to perform this check.

## Coverage improved and consolidation

| Module | Main improvements | Consolidation / migration |
| --- | --- | --- |
| Graphs | Interpreter bindings/defaults and cycles; canonical remapping/fusion; register liveness, operand order, signed zero, mixed native roots; analytical and higher-order derivatives; defensive Parameter regressions | Replaced deleted compiler/interpreter/cloning APIs. Retained snapshot and derivative intent; removed obsolete packed-array and wrapper-initial-value assumptions |
| Elements/base | Inherited annotation/metadata discovery, ClassVar exclusion, class-cache isolation, stable-handle failure atomicity, constraint isolation, required dependency validation, optical matrices and planar/GRIN limit derivatives | Replaced deleted discovery fixtures, renamed constraints, combined five narrower duplicate cases; independent physical checks remain |
| Composites | Independent Slab/ThickLens/Telescope/FourF aggregate optics; nested traversal/lengths; bindings and conflicts; shared parameters/children and constraint isolation | Replaced repr-only coverage and broadened discovery exclusion cases; preserved child wiring regressions |
| Inputs | Node-valued bindings and repair, numerical subclasses, inherited dataclasses, scalar tracer preservation, Gaussian properties/envelope/overlap and derivatives | Updated removed imports and mapping-only conversion assumptions; kept all Ray/RayBundle physics and derivative checks |
| System | Atomic registration, recursive constraint/binding collection, source-led media and continuity, canonical theta and local constraints, generated gaps, external-variable rejection, fixed infeasibility, snapshot isolation | Consolidated 100 functions into 60 focused functions; removed stale constructor/metadata APIs, preserving insertion-contract regressions |
| Simulation | Current constructor, numerical reconstruction, editable guesses, indexed single-pass propagation, Result/Stage invariants and errors, complex/float32 roots, analytical derivatives/JIT | Split the old monolithic suite into execution, initial-value and result tests; removed deliberately deleted Simulation.require/context APIs |

Test functions changed from 456 to 408, distributed across 21 files rather than
17. Parameterization accounts for the larger case count. Counts are descriptive;
the purpose of consolidation was clearer contracts, not a target test count.

## Failures requiring later investigation

### Graphs: invalid mutated Parameters are accepted (28 cases, both backends)

Compilation no longer defensively validates finite declared bounds and the
values inside them. Tests preserve the prior safety contract for fixed and
variable Parameters mutated after construction. They cover NaN, infinity,
out-of-range values, invalid bound types, NaN bounds and inverted intervals.
An infinite fixed optical radius remains explicitly allowed in a separate test.

Minimal case: construct p=Parameter(0.5,lower_bound=0,upper_bound=1), then set
p.value=2 and compile it. Expected ValueError; actual compilation succeeds.
This is missing implementation validation, not an LP or test-setup issue.

### Graphs: canonical parameter remapping fails (one case, both backends)

`remap_parameters(precompile((p,)), {id(p): 0})` for a variable p raises
`KeyError: False`. The membership comparison is assigned to parameter_id by a
walrus expression. Expected: a numerically equivalent canonical program.
Impact: local-to-canonical constraint alignment is unusable for ordinary inputs.

### Elements: optional-index exclusion hides a required empty dependency

One case fails under both backends. Reuse one empty InputNode as the optional
refractive_index and a required matrix entry. `validate()` filters out both
aliases and succeeds. Expected UnresolvedInputError for the matrix dependency.
This has been confirmed at runtime after the import repairs.

### System: insertion semantics conflict (six cases, both backends)

The earlier explicit API contract promised normal Python list insertion.
Negative and oversized indices should select/clamp insertion positions while
preserving multiple-element order. Current add() rejects them with ValueError.
The tests retain indices -1, -100 and 100, each with one and two additions.
This requires an explicit policy decision or implementation fix; the tests were
not rewritten to silently legitimize the changed rejection behavior.

### Autograd: fourth-order stacked scalar outputs (two cases only)

The scalar real/complex stacked-root cases raise `TypeError: first operand must
be array` in Autograd's sparse adjoint. The algebraically expanded independent
cost differentiates successfully. Crucially, the same failure also reproduces
in a numerical expression using xp.stack **without any Program**. Consuming
Program outputs as a native tuple sum differentiates successfully. This is an
Autograd stacking-path limitation, not a demonstrated compiler bug. All JAX
variants and the array-coefficient fourth derivative pass. Earlier diagnostic
claims that the stacked numerical reference passed were corrected in the graph
report and test reference before final verification.

## Whole-repository status

Normal `pytest tests -q --tb=line -p no:cacheprovider` was attempted under both
backends. Both stop with **14 collection errors** outside the reviewed unit
folders, including legacy helpers importing compile_ast/ParameterInfo and a
frozen dataclass inheriting the new non-frozen input base. Those tests and shared
helpers were not changed because their ownership and Solver migration were
outside this pass. The focused results above are not a whole-suite pass.

## Import-only repairs and remaining migration issues

Changed imports in graph/program_utils.py; simulation/__init__.py and
simulation.py; system/__init__.py; solver/constraint.py, solver.py and solution.py.
Removed stale export imports, imported SimulationResult for run(), removed
imports of locally defined/deleted initial-value classes, added missing Hashable,
and deferred legacy Solver-only imports to their consumers/type checking.

This allows normal package import and the new unit tests. It does **not** migrate
Solver from compile_ast, graph.utils.ast_equal or old Simulation metadata. Those
methods will still fail when exercised; replacing their architecture was not
authorized here. Simulation.run's old deferred return annotation and the old
type-only ParameterInfo reference also remain migration work.

## Remaining coverage gaps and ambiguous policy

- Full Solver/constraint-program alignment and end-to-end optimization need the
  pending Solver migration; no numerical optimization backend was invoked here.
- Interior probing and plotting have not been adapted to the new Result API.
  Numeric-z at() still references an absent _probe; no sampling policy was invented.
- Clarify explicit source n=None reconstruction versus ambient inheritance.
- Clarify binding-provider identity across different nested composites, cyclic
  containment, and changes to composition after construction.
- Clarify mutation guarantees for arrays exposed by successive returned results;
  build isolation from the original declarative source is covered.
- Sparse runtime-index remapping, repeated optimization of an already allocated
  Program, generic Node-containing container annotations, bundle-factory invalid
  domains and empty-bundle statistics remain unspecified rather than guessed.

## Detailed specifications and module reports

| Module | Behavioral specification | Test report |
| --- | --- | --- |
| Graphs | [graphs.md](graphs.md) | [graphs-results.md](graphs-results.md) |
| Elements/base | [elements.md](elements.md) | [elements-results.md](elements-results.md) |
| Composites | [composites.md](composites.md) | [composites-results.md](composites-results.md) |
| Inputs | [inputs.md](inputs.md) | [inputs-results.md](inputs-results.md) |
| System | [system.md](system.md) | [system-results.md](system-results.md) |
| Simulation | [simulation.md](simulation.md) | [simulation-results.md](simulation-results.md) |
