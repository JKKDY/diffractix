# Graph unit-test results

## Changes

Seven graph test modules now target the current API. Existing construction and
sinc tests were migrated from compile_ast to compile_graph. Interpreter and
traversal tests now use the single binding map and optional id-to-value mapping.
Compiler tests retain snapshot, alias/cycle-repair, identity, bounded mutation,
native-output, and independent hybrid-gradient/Hessian contracts.

New focused modules cover compiler optimization and program remapping/fusion:
register reuse with repeated sources and retained early outputs, dead-code
removal, direct/duplicate outputs, CSE, signed zero, noncommutative operations,
nonfinite runtime arithmetic, ambiguous array equality, canonical parameter
reordering and extension, symbol slot reordering and fusion of allocated programs.
High-order checks compare against independently written numerical expressions,
not only another execution of the same compiler.

Removed obsolete coverage for deleted clone_ast/collect_variables APIs, the
CompiledAST wrapper, packed-array/sequence dual APIs and compile-time
parameter_snapshot arguments. Their relevant identity, mutation-isolation,
traversal and derivative behavior remains tested using Program. A new Program
does not expose initial_values; initial-guess ownership belongs to Simulation.
Runtime native-root arrays are not asserted to be owned packed-array copies.

No production, helper, dependency or test configuration files were changed.
No skip/xfail markers or compatibility import shims were added.

## Execution

The initial ordinary run stopped on a stale package import. Diagnostic runs
using pytest's --continue-on-collection-errors provided partial-import-state
coverage, but those are now superseded by normal clean-process runs after the
parent agent's user-authorized import-only repairs. This graph agent did not
modify production.

All runs used `.venv/Scripts/python.exe -m pytest tests/unit/graph -q
--tb=short -p no:cacheprovider` in separate backend processes. No packages were
installed and no import shims were used.

| Backend | Passed | Failed | Collection errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Autograd, normal clean-process run | 200 | 31 | 0 | 0 |
| JAX, normal clean-process run | 202 | 29 | 0 | 0 |

All seven test modules passed syntax validation. The earlier partial diagnostic
orders also exercised all 231 distinct cases with the same aggregate counts,
but are not used as the final clean-process verification.

## Failures and minimal reproductions

### Canonical parameter remapping: both backends

`test_parameter_remapping_reorders_and_extends_theta_including_direct_outputs`
raises `KeyError: False` instead of preserving numerical evaluation. Minimal
reproducer:

```python
p = Parameter(1.0).variable()
program = precompile((p,))
remap_parameters(program, {id(p): 0})
```

The walrus expression in program_utils.remap_parameters assigns the result of
`old_id not in parameter_indices` to parameter_id. Expected: remapped program
reads p's canonical slot. Actual: lookup under False. Impact: any normal local
to canonical theta alignment can fail before constraint fusion/evaluation.

### Defensive Parameter validation: 28 cases per backend

Compilation accepts directly mutated invalid values or bounds. Minimal examples:

```python
p = Parameter(0.5, name="validated", lower_bound=0.0, upper_bound=1.0)
p.value = 2.0
compile_graph((p,))  # expected ValueError, currently accepted

p = Parameter(0.5, lower_bound=0.0, upper_bound=1.0)
p.upper_bound = "bad"
compile_graph((p,))  # expected TypeError, currently accepted
```

The preserved regressions cover fixed and variable Parameters; NaN, +/-Inf and
out-of-range bounded values; bool/nonreal/NaN/inverted bounds. Both precompile
and compile_graph are checked for value mutations. These are existing safety
contracts from the previous compiler, not a ban on legitimate infinite fixed
optical radii: a separate flat-mirror-style expression using radius=inf passes.

### Fourth-order scalar stacking: Autograd limitation, two cases

The real and complex fourth-order variants fail with `TypeError: first operand
must be array` in Autograd's sparse_add/getitem adjoint. First, second, third
and nested-Hessian checks pass; the array-coefficient fourth derivative passes.
The same fourth-order failure occurs in an independently written numerical
expression using `xp.stack((r, r+1, r**2))`, with no Program or compiler at all.
A native-tuple Program consumed without stacking passes the fourth derivative.
This is therefore an Autograd stacking/adjoint limitation, not an established
compiler defect or evidence of lost differentiation in native tuple execution.

The original version of this report incorrectly said the stacked analytical
reference passed. A clean-process recheck showed the exception actually occurred
while constructing that reference. The test now uses the independently expanded
cost `|r|**4 + 2*|r|**2 + 2*Re(r) + 1` for its reference, without stacked-root
adjoints; that reference passes fourth differentiation. The test still checks
the Program's stacked-root path and retains the two failures, with no skips or
weakened derivative expectations.

The roots are `r=p**3 + Symbol("coefficient")*p**2`, `r+1`, and `r**2`.
Execution receives `theta[:1]` plus a coefficient derived from the remaining
theta entries. Minimal backend-only reproduction:

```python
def cost(t):
    r = t[0]**3 + (t[1]**2 + t[2]**3)*t[0]**2
    values = xp.stack((r, r+1, r**2))
    return xp.sum(xp.real(values*xp.conj(values)))

fourth = cost
for _ in range(4):
    fourth = jacobian(fourth)
fourth(xp.array([0.4, 0.2, 0.7]))
```

Expected: the finite fourth derivative of this polynomial. Actual: TypeError
under installed Autograd. The normal clean-process JAX run passes all
corresponding derivative cases.

## Remaining uncertainties

- Import-only repairs now allow normal graph collection. Solver/System migration
  behavior remains outside this module; the graph report does not certify it.
- Sparse/nonconsecutive symbol remapping policy and optimizing an already
  register-allocated Program are not documented; tests do not invent them.
- Nonfinite unbounded Parameters have optical exceptions. Defensive tests
  intentionally use finite declared bounds where invalidity is unambiguous.
- Active tracer-to-literal capture retains existing numerical-backend-specific
  coverage outside this owned folder; no broad subprocess infrastructure added.
