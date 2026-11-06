# System unit-test results

## Changes

Rewrote the two owned System unit-test files around the current API and build
passes. The former 100 test functions are now 60 focused functions, producing
121 parametrized test cases. No production file, dependency, helper or test
configuration was changed by this agent.

The rewrite preserves independent layout arithmetic, element identity, nested
paths, medium continuity, graph snapshots, source snapshots and canonical theta
ordering. Repeated getter-only and per-field metadata tests were combined into
tests that evaluate the compiled output and check the resulting metadata.

Removed obsolete assumptions deliberately superseded by the refactor:

- Positional ambient constructor arguments and `ambient_n_variable` are replaced
  by the variadic constructor and a supplied variable Parameter.
- Positional z, `add_input_beam`, `add_context`, `compile_context`, and
  `requirements` tests use keyword z, `add_input`, `bind`, `bindings`, and
  `constraints` respectively.
- Removed `_compile`, ParameterInfo, fixed-Parameter metadata and location_map
  expectations now test `_compile_optics`, variable-only ParameterData,
  ElementData identity/positions and indexed optical outputs.
- The old separate fixed-continuity validator tests now exercise resolution's
  fixed mismatch errors and variable equality generation directly. They do not
  require variable continuity to be initially satisfied.

The original Python-list insertion boundary expectations remain unsuppressed;
the rewrite does not change them to match the current range rejection.

## Coverage strengthened

- Atomic registration of nested batches, invalid types, malformed indices,
  malformed absolute positions and structurally equal dataclass elements.
- Independent Symbol default copies, variable ambient identity and safe ambient
  rebinding, read-only binding views, complex constants, per-source/element
  bindings, repeated provider identity, duplicate binding errors and wavelength
  collision rejection.
- System/input/composite/child constraint order, identity deduplication and
  aggregated validation diagnostics with call-site information.
- Layout arithmetic for multiple absolute positions; dependent gaps tracking
  both target position and upstream length; generated Space metadata and
  nonnegative-length constraints, including later infeasible theta values.
- Source-led refractive-index resolution, ambient fallback, inherited fields
  left untouched, Interface medium chains, fixed mismatches, variable initially
  mismatched equality constraints and Symbol-bound index variables.
- Shared Parameter deduplication, deterministic canonical order, unused binding
  exclusion, explicit continuity-only dependency extension, fixed/derived
  inspection output indices and repeated-element metadata.
- Compilation isolation from later live values, variable flags, bounds,
  InputNode replacements, expression mutation and binding reassignment.
- Owned ray-bundle array snapshots and source reconstruction; mixed complex
  aperture coefficients preserve real source wavelength and correct local n.
- Constant constraints: satisfied boundaries omitted; violated constants reported
  unsatisfiable; NaN/Inf rejected; variable constraints may begin unsatisfied.
- External variable rejection directly and through Symbol bindings; fixed
  external values allowed; unresolved Symbols rejected even with defaults.
- Graph/deferred-callable declaration interleaving, bounds/labels, no context
  tracing during System compilation, zero-argument normalization exactly once,
  local constraint input order and nonmutation of the canonical mapping.
- Repeat-build constraint stability and malformed System/element attachments
  failing during build.

## Execution

After the parent repaired the production imports, normal pytest collection and
execution succeeded. No package shims or compatibility aliases were used.

| Backend | Passed | Failed | Skipped | Warnings |
| --- | ---: | ---: | ---: | ---: |
| Autograd | 115 | 6 | 0 | 0 |
| JAX | 115 | 6 | 0 | 0 |

Commands, each in a fresh process with its backend environment variable:

```text
.venv/Scripts/python.exe -m pytest tests/unit/system -q --tb=line -p no:cacheprovider
.venv/Scripts/python.exe -m compileall -q tests/unit/system
git diff --check -- tests/unit/system
```

Syntax validation and the scoped whitespace check pass. Git reports only its
normal LF-to-CRLF working-copy notice.

## Confirmed failures

All six are instances of
`test_indexed_insertion_preserves_python_list_semantics`, using indices -1,
-100 and 100 with one or two inserted elements. Valid indices 0 and 1 pass.

Minimal reproducer:

```python
system = System(Plane(), Plane(), Plane())
element = ThinLens(f=0.1)
system.add(element, index=-1)
```

Expected from the earlier explicit public API requirement: insert immediately
before the final element, following Python list insertion semantics. Oversized
positive indices append, and sufficiently negative indices insert at the start;
multiple supplied elements remain consecutive and in order.

Actual: `System.add()` raises `ValueError("Insertion index out of range.")` at
`src/diffractix/system/system.py:139` for all three index values. This is a public
contract conflict, not a numerical/backend limitation or test setup failure.
The tests are retained so the policy can be settled explicitly. No production
fix was attempted.

## Remaining gaps and open policy

- A source whose explicit n is None uses ambient during resolution but may
  reconstruct numerically with None. This suite verifies resolution only; a
  complete source/run policy remains to be decided with Simulation.
- Shared binding providers under different nested composites may collide;
  recursive alias policy is not yet specified. Straightforward direct identity
  deduplication and distinct-provider collisions are covered.
- Deferred Python callable closure mutation is intentionally the user's
  responsibility. These tests ensure deferral/order, not frozen closure values.
- Constraint program alignment/fusion with the Solver remains a separate
  migration task. Tests explicitly distinguish local and canonical ordering.
- Full optimization, arbitrary-z probing and plotting are outside System unit
  ownership. No optimizer backend or solve() was invoked.
