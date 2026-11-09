# QA handoff and current contracts

Updated 9 November 2026. This is the current QA baseline and policy register.
API references and behavioral specifications describe the same contracts; superseded task reports were removed.

## Current baseline

Latest recorded complete runs: Autograd **1338 passed, 4 skipped**; JAX **1340
passed, 2 skipped**, without failures or warnings. IPOPT/NLopt native packages
were absent. Autograd additionally skips two JAX-only checks. Explicit fourth-order
cases were removed by scope decision; this is not an Autograd fourth-order fix.

Record the tested commit and any working-tree changes when reproducing these
results. A passing local test run is not evidence that untested examples, wheels
or CI matrix cells pass.

## API and QA scope

Current names: System(optical_input=...), result[element].before/after,
result[element, occurrence], stage.z_start/z_end, solution.value(parameter).
No legacy System(input=...), ctx.at(), Solution indexing or value_of() contract.
The README quick start uses the current API. Examples and wheel/example scripts
still need migration and release execution; their historical APIs do not define
the current library contract. Record those as release findings; do not use them as
an oracle to restore old APIs. References in docs describe the rewritten API.

Use public APIs for user workflows. Focused white-box unit tests are appropriate
for compiler remapping, Program reduction and callback identity/liveness contracts.
They should check mathematical behavior rather than private instruction numbering
or require one arbitrary implementation layout.

Default QA scope is tests and reports. Import-only library repairs require a clear
need and explicit reporting; substantive library fixes require maintainer approval
unless a particular fix has already been authorized. Do not install dependencies,
change configuration or remove legitimate failures to obtain a green result.
The objective is release readiness: assess behavior and test quality, find bugs,
and report them independently from test/setup migration problems.

Package metadata permits Python >=3.11. CI/release validation currently covers
3.11-3.13; 3.14 has successful local runs but is not yet a CI-backed release claim.
Do not expand the CI/support matrix merely from those local results.

## Validation boundaries

System constructor placements and direct add() calls record their user call site,
so build errors point to the design code. Parameter construction rejects NaN
immediately, while valid infinite optical radii remain supported. Compilation
still checks state mutated after construction.

GaussianBeam construction and w_R_from_q reject nonfinite q and imaginary parts
that are zero or negative, before division. Numerical subclasses use the same
constructor check, including reconstruction at build. Valid small positive
imaginary parts have no arbitrary epsilon cutoff. Differentiation/JIT follows
backend.require: concrete invalid values raise, dynamically traced predicates
remain unchecked inside compiled execution.

## Snapshot and binding policy

Compiled Programs capture expression structure/fixed values. System snapshots
ordinary mutable source fields and numerical mappings at build. Later edits to
external Parameters or source arrays must not alter built numerical behavior.
Simulation.bindings is a shallow mapping snapshot: AST objects inside it are not
deep-cloned. Because new Solver expressions may resolve them later, this remains
a documented limitation/TODO. Do not claim full binding-AST snapshot isolation.
Python callable captures remain the user's responsibility, especially deferred
control-flow callables. A deferred callable must return numerical values.

Binding deduplication across different nested composite branches is still an open
policy question. Current System/composite collectors deduplicate direct provider
identities locally and reject duplicate keys, even for equal values. They do not
share one recursive identity registry across all branches. Do not invent a
cross-branch guarantee. A reasonable future policy is one collection per provider
identity, while distinct providers with the same key still conflict; this has not
been approved or implemented as a new contract.

## Interface boundaries and cleanup

- Numeric-z result.at() is removed entirely. Use indexed stage boundary views.
  Arbitrary interior probing is not supported; no _probe path remains.
- ABCD with an explicit index participates in continuity like other non-Interface
  elements. It must match the incoming medium (or constrain variables to match).
  Interface alone declares a refractive transition. No explicit ABCD index means
  inheritance of the current medium.
- solver/context.py is unused by the rewritten pipeline but remains exported.
  Removing it and its exports is a separate small cleanup, not a new API to test.
- Fourth-order Autograd behavior is outside current release verification. Current
  optimizers use first/second derivatives. No workaround or blanket support promise
  is required; representative nested differentiation tests remain.

## Objective, plotting and RayBundle details

Value matching residual is exactly (value(result)-target)/scale. Cost sums
weight * sum(square(residual)) across objectives. Scale defaults to one, cannot
be zero, and target zero is valid. The matching shorthand has no weight keyword;
use an explicit Objective to set a nondefault weight.

Public plotting: result.plot(...) and diffractix.plotting.plot(result,...), both
returning (figure, axes). Neither calls show() or save. Concrete Ray, RayBundle
and GaussianBeam are supported, up to 41 displayed bundle rays, with z in metres
and transverse coordinates in millimetres. Only Spaces get continuous sampling;
other elements retain boundary states. Axes, labels and show_elements/show_axis
controls are supported. See system-simulation-result.md.

RayBundle enforces matching one-dimensional x/theta arrays; empty bundles and
single rays are supported for construction/propagation. Factories currently use
backend linspace, without explicit count/radius-domain validation. Empty-bundle
statistics delegate to backend mean/std (typically NaN with backend-dependent
warnings); there is no settled exception/warning contract. Do not demand specific
warnings or successful finite statistics. Factory-domain and empty-statistics policy requires a separate decision before
new validation assertions or library checks are added.
