# Integration test contracts

Read [System, Simulation and Result](../system-simulation-result.md),
[the graph reference](../graph-reference.md) and [QA status](../qa-status.md).
Tests exercise current public workflows across packages, with small deterministic
models and independently derived optical or optimization expectations.

## Optical pipelines

Exercise System -> Program -> Simulation -> SimulationResult for Ray, RayBundle,
GaussianBeam and custom states/elements. Include nested composites, Slab/ThickLens,
material transitions, repeated elements, generated gaps and dependent parameters.
Verify physical-angle ABCD equations, local refractive indices, complex roots,
representative float32 behavior, source variables, canonical theta sharing and
snapshot isolation from declarative objects. Test representative derivatives/JIT.

Bad paths include invalid models/layouts, unresolved optical Symbols, external
variable Parameters in constraints, fixed infeasible constraints, missing elements
and ambiguous or invalid occurrences. Do not duplicate every unit validation case.

## Inverse design and lifecycle

Use actual small SciPy solves, including SLSQP and a representative trust-constr
case. Check physical optima, feasibility and complete Solution reporting, not
iteration counts. Include source-plus-element design, Gaussian waist matching,
RayBundle array residual focus, continuity, constraint collection, coupled gaps,
repeated-occurrence result Symbols and vector constraints.

Exercise eager/JIT behavior and numerical dynamic fallback. Deferred callables
must return numerical outputs; JAX-incompatible branches must fail when explicitly
JIT compiled. Check infeasible/iteration-limit termination separately from
feasibility, recovery after preparation errors, changed initial guesses/bounds,
constraint restoration, fixed-coordinate unscaling and prior Solution isolation.
A new Solver(solution.simulation) must retain constraints and canonical IDs.

## Backends and parity

Test real native adapters with active constraints whose omission changes the
optimum. Skip IPOPT/NLopt only when their packages are absent. Shared small
calculations run in isolated Autograd/JAX processes and compare values,
gradients/Jacobians/Hessians and real solve results against independent equations.
Do not replace real convergence with mock adapters or use large example benchmarks
as routine tests. Example/wheel release scripts remain separate checks.

## Plotting

Use the public result.plot() and diffractix.plotting.plot(result) APIs with Agg.
Check actual data, user axes preservation, no implicit show/save and no mutation.
Space intervals have continuous samples; GRIN/other thick elements preserve
boundary states without invented interior propagation. Include empty paths and
Ray/Bundle/Gaussian results. Avoid pixel/style snapshots.

## Execution and limits

Run the complete integration directory under each installed numerical backend in
separate processes. Retain legitimate failures; report library bugs, stale tests
and policy questions separately. No xfail or arbitrary skip to obtain green runs.
Optional-library/JAX-specific availability skips remain legitimate.

No arbitrary-z probing, multi-input model, deep immutable binding ASTs or arbitrary
Python closure snapshot is promised. First/second derivatives and representative
nested transformations are covered; explicit fourth-order support is outside the
release scope. No global convergence guarantee is made for arbitrary nonconvex
models. Keep fixtures local and tests concise; avoid a broad Cartesian product.
