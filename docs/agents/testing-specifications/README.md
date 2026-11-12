# Test specifications

These files describe behavioral contracts for current tests. They are maintained
references, not task assignments or historical test reports. For current results,
release gaps and unresolved policies, see [QA status](../qa-status.md).

| Area | Specification | Tests |
| --- | --- | --- |
| Graph | [Graphs](graphs.md) | tests/unit/graph |
| Shared base and elements | [Elements](elements.md) | tests/unit/core, tests/unit/elements |
| Composites | [Composites](composites.md) | tests/unit/composites |
| Inputs | [Inputs](inputs.md) | tests/unit/states |
| System | [System](system.md) | tests/unit/system |
| Simulation and results | [Simulation](simulation.md) | tests/unit/simulation |
| Solver package | [Solver](solver.md) | tests/unit/solver |
| Cross-package workflows | [Integration](integration.md) | tests/integration |

## Test quality

Use independent analytical or physical expectations. Preserve regression tests
and meaningful error paths. Public APIs are preferred for workflows; private
passes may be tested to isolate compiler, reduction or callback contracts. Avoid
asserting private register numbers, exact formatting or incidental object layouts.
Do not add duplicate matrices when one end-to-end scenario establishes the missing
handoff. Node equality is symbolic: compare identities with is or IDs.

Run Autograd and JAX in separate processes using DIFFRACTIX_BACKEND. Shared tests
use diffractix.backend for differentiation. Isolated JAX-specific tests may import
JAX. Missing optional packages justify targeted skips; real failures do not.
Controlled native API substitutes are appropriate in adapter unit tests; real
optimizer integration tests must call installed native libraries.

Do not weaken legitimate failures or silently change library behavior to satisfy
tests. Report setup errors, library defects and unsettled policies separately.
No new dependencies or testing framework are required. Example and wheel checks
belong to the existing scripts, not duplicate pytest subprocess wrappers.
