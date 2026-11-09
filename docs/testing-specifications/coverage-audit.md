# Independent coverage audit

8 November 2026. Read-only review of current source, unit and integration suites,
solver/integration behavioral specifications and their result reports. No source
or tests changed; no additional test runs. This is a behavioral audit, not a line
coverage score. Candidates below require selection, not wholesale implementation.

## Assessment

The suite already has strong progressive integration: graph-to-propagation,
source-and-element theta sharing, traced runtime result Symbols, derivatives and
JIT, and actual small optical solves. The real gaps are a few public workflows and
representation boundaries. Another broad happy/bad-path matrix would mainly
repeat existing checks. Prioritize 3-5 compact scenarios or strengthen existing
weak native smoke tests; avoid a new test framework.

## Highest-value candidates

### A. Reuse Solution.simulation as an optimization model

Priority: recommended before release. Cost: one small two-stage solve.

Current public Solution.simulation is documented as a complete updated simulation,
with solved defaults, current bounds and full original constraints. Existing
unit test_solution.test_updated_simulation_and_canonical_value_lookup and
integration test_solver_solution.test_bound_relaxation_restores_constraints_and_keeps_earlier_solution
only run or inspect it. None constructs a fresh Solver from the returned Simulation.

Proposed scenario: solve a Ray/Space distance model with an explicit upper-limit
constraint that the first solve's tight parameter bounds make redundant. Construct
Solver(first.simulation), add a different objective that requests a longer gap,
and relax the bound on this new solve. Expect the retained explicit constraint to
become active and cap the second physical optimum. Original solver defaults and
first Solution remain unchanged. Independently, final ray x = x0 + d*theta.

Why distinct: this crosses Solution -> Simulation -> Solver -> reduction -> native
backend; current repeated-solve tests stay inside one original Solver. It catches
lost full constraints, wrong canonical IDs, stale metadata/bounds and bad offset
fusion on a public reuse path. Avoid merely checking identity/property presence.

### B. A real RayBundle vector-residual inverse design

Priority: recommended before release. Cost: one small solve, eager/JIT if cheap.

Existing test_ray_bundle compares bundle propagation to independent scalar rays.
test_solver_jit.test_vector_optical_constraints_and_scalar_target_solve_together
uses Ray.final.vector (two scalar state properties). No actual inverse-design test
uses a bundle's array-valued optical state as an objective/constraint.

Proposed scenario: parallel bundle x=(-r,0,r), theta=zeros, ThinLens(f=variable),
Space(d=L). Target result.final.x (an array). Analytically x_out=x_in*(1-L/f), so
f=L uniquely minimizes nonzero rays' residuals. Check optimized focal length,
vector residual shape and values, objective cost and each ray's output against
that formula. A componentwise bound can be included only if it meaningfully
changes the optimum; do not add an inactive constraint merely for appearance.

Why distinct: combines a genuinely vectorized state, native array-valued Symbol,
objective sum/flattening and real optimizer. It is not another scalar Ray solve
nor another generic Literal-array callback test.

### C. Strengthen optional native backend smoke tests to have active constraints

Priority: recommended before release. Cost: modify existing tests, no new matrix.

In tests/integration/test_backends.py, native IPOPT/NLopt smoke tests both expect
[2,1], the unconstrained minimum of constrained_problem(). Because that point is
feasible, these tests can pass even if native constraint registration is ignored.
The unit fake-adapter matrix is thorough, but cannot establish actual native API
behavior. SciPy already has an active mixed equality/inequality solve.

Use the same active-constraint projection already used for SciPy (quadratic centered
at [2,1], x+y>=4, x-y==1, finite box), optimum [2.5,1.5], cost 0.5. Keep native
package availability skips. Check physical feasibility and optimum, not exact
iterations/messages. If maintaining a one-constraint smoke is preferable, change
its bound to x+y>=4 so optimum [2.5,1.5] still requires enforcing the constraint.

Why distinct: improving the oracle of existing conditional tests, not adding more
mocked adapter or optional-package import checks. Packages are absent locally;
these cases cannot be validated against actual libraries until installed elsewhere.

### D. Plotting must not invent propagation inside a GRIN/unknown thick element

Priority: recommended before release. Cost: one small sampling/plot test.

Current test_plotting checks public axes behavior for Ray/Bundle/Gaussian, free-space
ray segments and empty paths. The removed legacy suite contained more broad plotting
coverage. Current sampling source explicitly promises that no segment crosses an
unknown thick element; renderer labels that interval as boundary states only.

Build Space -> nonzero GRIN -> Space with a Ray or Gaussian state. Assert sampled
segment z ranges cover only the two Spaces, neither segment crosses the GRIN
interior, and stored boundary points match the actual SimulationResult. Optionally
verify the renderer exposes a boundary-only interval label, without asserting exact
colors, artists count or text placement.

Why distinct: plotting cannot correctly infer GRIN interior states from boundary
ABCD values. This is a scientific meaning contract; existing thin-lens/free-space
plots do not exercise it. One test can cover a real unknown-thick built-in; no
custom-element matrix required.

## Lower-priority candidates (select at most one or defer)

### E. Feasibility-only actual solve

No actual integration solve appears to omit every objective while retaining a
constraint. Unit test_execution_regressions tests the zero floating objective and
its derivatives, not native behavior. Use a ray source position initially outside
an achievable output lower bound, with no target(). Expect success/feasible,
cost==0, objectives==(), constraint satisfied. Do not assert a unique optimum:
there is none. Cheap but less important than A/B/C; may be worth one scenario.

### F. Fixed-coordinate bound overrides through native SciPy

Equal x_lower/x_upper is explicitly allowed and checked by test_cache's captured
Problem tests. Actual solves currently vary ordinary intervals only. One
partially fixed two-variable Ray source solve would check SciPy's fixed-variable
projection followed by physical unscaling and canonical Solution.value lookup.
Expected free coordinate from a linear ray equation; fixed coordinate exactly at
its bound. This catches adapter behavior unit callbacks cannot model. Prefer a
partially fixed problem; all-fixed method semantics are less clear and should not
be invented. Defer if test budget is tight.

### G. Wavelength -> GaussianAperture -> Solver -> Solution

Source wavelength theta is tested in propagation/snapshots; GaussianAperture
complex matrix and radius derivatives are tested; real Gaussian optimization
currently varies a Space distance. There is no actual solve varying source
wavelength through aperture's wavelength Symbol. With fixed q=i*zR and aperture a,
w_out^2 = 1 / (pi*n/(lambda*zR) + 1/a^2), which gives an independent positive
wavelength optimum for a chosen reachable target width. A single well-scaled solve
would close that binding/complex propagation/runtime state loop. However existing
pieces are already well covered; defer behind A/B unless source spectral design
is an important first-release workflow. Do not optimize q or complex Parameters.

## Already covered: do not add more of these

- Source angle + lens focal joint design; Gaussian waist through actual SLSQP and
  trust-constr; source and element fields sharing one canonical Parameter.
- System/composite/child/solver constraint collection; independent variable index
  continuity; repeated occurrence state/z Symbols; generated-gap nonnegative length.
- Small coupled optical gaps with a known constrained optimum at metre scales.
- Same-Solver repeated guesses/bounds; restored reduced constraints; immutable old
  Solution; updated Simulation metadata; callback/JAX executable reuse.
- Vector constraints and weighted multiplier/Hessian remapping: extensive units
  and real Ray-vector constrained solve already exist.
- Dynamic numeric Python branch eager solve, explicit JAX JIT failure, runtime
  Node-return rejection, user exception propagation and retry.
- NaN/Inf objective/constraint validation, shape stability, invalid x0/bounds,
  wrong model types, option forwarding/copying, fixed-model lifecycle: units cover
  these independently. Repeating each through native optimization adds little.
- Snapshot arrays and Parameters after build/solve, custom dataclass state/element
  compilation, complex roots, float32 arithmetic, planar-interface derivatives,
  Gaussian aperture local medium and repeated-element interpretation.
- Plot existing axes, no implicit show, empty trace and metre-to-mm ray samples.
- Reducer mathematical proofs/edge cases, strict nextafter bounds, empty/vector
  outputs, equality rank, unbounded domains and signed zero: do not duplicate the
  Program unit matrix as a larger integration suite.
- Small graph/Gaussian/real solve backend parity; optional import/error contracts.

## Existing problems and policy questions are not new missing tests

- Historical 39 backend-unit setup errors come from the shared legacy representative
  fixture importing compile_ast. One additional scalar-literal unit test does the
  same. Repairing that migration restores already-written coverage; adding new
  tests around it would obscure the actual issue.
- Fourth-order Autograd stacking has known retained regressions and is explicitly
  deferred. Do not add another reproducer or xfail.
- Plotting's element_info rename issue is now fixed and the five integration cases
  pass. This audit does not identify a further confirmed production bug.
- Binding ASTs and deferred callable captures have an explicit unresolved/shallow
  snapshot policy. Do not assert full deep immutability without a policy decision.
- No multi-input or arbitrary interior-z probing is promised. Numerical input
  System rejection, standard API abuse and malformed parameter state already have
  focused unit coverage; do not test unsupported workflows as if supported.
- A nonfinite optimum, absurdly ill-conditioned model or arbitrary nonconvex global
  optimum does not justify inventing solver convergence promises.

## Full release pipelines and examples

CI already runs scripts/run_examples.py under both backends and an isolated wheel
smoke script. The examples visibly still call old APIs such as System(input=...),
ctx.at()/after(), solution.targets and simulation.source. They are a known migration
issue by inspection. Prefer updating and executing the existing runner rather than
adding pytest subprocess wrappers that repeat it. Current small integration tests
do not establish that advanced multi-station examples converge; that is a separate,
more expensive release verification task after example migration.

No additional dependency installation, fuzzing framework, subprocess-per-case
matrix, coverage-percentage target or broad cross-product of every state/optimizer
is justified for this first release.
