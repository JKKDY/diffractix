# Solver integration coverage

8 November 2026. Tests only; no production, helper, configuration or unit changes.

## Scenarios

- Joint source-angle/lens-focal optimization recovers an analytical ray design,
  eager and JIT, with physical-coordinate values and reported residual costs.
- Gaussian waist position through real propagation, both SLSQP and trust-constr,
  checked against the analytical complex q and waist radius.
- System, composite, child-element and explicit solver constraints compose and
  change an actual Slab design optimum; reporting preserves declaration order.
- Independent variable refractive indices optimize together while continuity is
  enforced; reused Space occurrences trace both longitudinal positions and states.
- Generated-gap inequalities prevent crossing an absolute station; coupled gaps
  at millimetre scale retain the required output clearance.
- Repeated real solves relax parameter bounds and restore needed constraints;
  core callbacks are reused and previous solutions/defaults remain isolated.
- Vector optical constraints and scalar objectives solve together with correct
  residual shapes and full reporting; reduced-out constraints remain reported.
- Real inconsistent runtime constraints return an unsuccessful, infeasible
  solution; native iteration-limit termination remains independently feasible.
- Eager numerical branch fallback solves; JAX-dependent Python branches fail
  explicitly under JIT; deferred symbolic Node returns are rejected.
- User preparation exceptions propagate and a subsequent corrected call can retry.
- Built requirement programs retain original element handles after external
  reassignment; source/model mutation cannot change an existing solved result.
- Bound compile expressions and aliases compose with runtime result Symbols;
  external variable Parameters and missing bindings fail before optimization.

## Consolidation

Replaced obsolete test_inverse_design.py, test_solver_solution.py,
test_solver_jit.py, test_snapshot_lifecycle.py and test_symbol_resolution.py with
focused current-API scenarios. Removed test_solver_compilation.py and
 test_solver_expression_evaluation.py: these depended on removed private compiler,
legacy SolverContext and performance representations. Their relevant preparation,
canonical mapping, derivative, shape, dtype and fallback contracts are covered
by current solver units plus the end-to-end scenarios above. No production aliases
or legacy APIs were restored. Removed old tests that demanded deferred Node
interpretation, Solution.value_of/arbitrary AST evaluation, Simulation.require,
old source/ambient continuity policy, and unsnapshotted mutable binding AST policy.

## Runs

Owned files: test_inverse_design.py, test_solver_solution.py, test_solver_jit.py,
test_snapshot_lifecycle.py and test_symbol_resolution.py.

- Autograd: 25 passed, 1 skipped, no warnings.
- JAX: 26 passed, no warnings.
- The only Autograd skip is JAX tracer-dependent Python-branch rejection.

No production bug was confirmed. Unsupported nested .q.real symbolic attribute
access was avoided by using GaussianBeam's public z0 property; adding complex graph
accessors is outside this test contract. Mutable AST bindings remain an explicit
snapshot-policy TODO, not a new guarantee. Large multi-station designs and native
IPOPT/NLopt convergence belong to separate optional/slow verification.
