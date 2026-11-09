# Selected coverage improvements

8 November 2026. Independent second review of coverage-audit.md, checked against
current Solution construction, native problem fixtures, optical integration tests,
solver units, and plotting sampling/rendering. No code changes or test runs.

## Decision

Add four concise scenarios (A, B, D, F) and strengthen two existing native smoke
tests (C). Prioritize A/B/C if budget is tight. No new testing framework or broad
state/optimizer/backend cross-product is warranted.

### A — Add: fresh Solver from Solution.simulation

Home: tests/integration/test_solver_solution.py.

Use a Ray->Space distance d model: initial d=0.03, bounds [0.01,0.3], source x=0,
angle=0.01. Explicitly require d<=0.15. First solve requests d=0.2 but overrides
upper bound to 0.1, making the explicit constraint redundant. Then construct
Solver(first.simulation), add a new target requesting d=0.25, relax the bound to
0.3, and solve again.

Independent oracle: final.x=0.01*d; first optimum d=0.1, second d=0.15 rather than
0.25. Check success/feasibility, constraint label retained, canonical Parameter
lookup, and first Solution/original Simulation defaults and bounds unchanged.

Existing test_bound_relaxation_restores_constraints_and_keeps_earlier_solution
reuses one Solver; it never tests this public Solution->Simulation->new Solver
handoff. This catches lost full constraints, canonical IDs or frozen metadata.
One eager SLSQP case under each numerical backend is enough; no JIT duplication.

### B — Add: real RayBundle array-residual optimization

Home: tests/integration/test_inverse_design.py.

Parallel bundle x=(-r,0,r), theta=zeros, n=1; ThinLens(f variable) then Space(L).
Use L=0.1, r=0.001, f initially 0.15 with positive bounds [0.05,0.3]. Target the
array residual result.final.x/r so small metre units do not cause premature stopping.

Independent oracle: x_out=x_in*(1-L/f), theta_out=-x_in/f. Unique optimum f=L.
Check f, final states, residual shape (3,), reported sum-of-squares cost, success
and feasibility. Do not add an inactive constraint just to expand apparent coverage.

Bundle integration currently propagates/differentiates arrays but does not solve;
the existing vector optical solve uses a single Ray.vector made of scalar state
properties. This tests actual native-array Symbol->cost->AD->optimizer composition.
Use eager/JIT under both installed backends; this small matrix is justified here.

### C — Strengthen existing optional native smoke tests

Home: tests/integration/test_backends.py, existing NLopt and IPOPT smoke tests.

Confirmed weak oracle: constrained_problem minimizes (x-2)^2+(y-1)^2 with x+y>=2;
current expected (2,1) is feasible even if every native constraint is ignored.
Fake-adapter units cannot establish real native registration.

Inside each existing test use replace(constrained_problem(),
constraint_lower=np.array([4.])). Projection optimum is (2.5,1.5), cost 0.5.
Assert status, physical optimum/cost and actual feasibility. Keep package absence
skips and existing approximately 1e-5 coordinate tolerance. Do not install packages.
The current active mixed SciPy integration already covers SciPy: add no duplicate.

No new test count or numerical-backend matrix is needed. The native Problems use
explicit NumPy derivatives. Actual optional verification remains conditional on
installed packages; do not claim these ran locally when skipped.

### D — Add: no invented trajectory inside a thick GRIN

Home: tests/integration/test_plotting.py.

Space(0.02)->GRIN(d=0.03,g=2,n=1)->Space(0.04), Ray with nonzero position/angle.
Sample and plot the actual SimulationResult. Exactly two homogeneous-space segments
span [0,0.02] and [0.05,0.09]; neither contains the GRIN interior (0.02,0.05).
Use x(z)=x_start+theta_start*(z-z_start) on each Space. Preserve GRIN boundary points;
optionally assert the boundary-only legend meaning, not artist styling/counts.

Current plotting tests cover free spaces/thin lens/empty systems, not unknown thick
interiors. The source explicitly promises this physical interpretation. One Ray
case under each backend suite process suffices; no custom/state/pixel matrix.

### F — Add after A–D: partially fixed coordinate through native SciPy

Home: tests/integration/test_solver_solution.py.

Ray x and theta are independent variable Parameters; Space(1). Initial x=0.2,
theta=0.1. Override x's lower and upper bounds to 0.2 and leave theta free in [-1,1].
Construct arrays in canonical index order. Target final.x-1.

Independent oracle: final.x=x+theta; x stays 0.2, theta reaches 0.8, cost is zero.
Check physical/canonical values, feasibility, updated Simulation defaults/bounds,
and unchanged original defaults/bounds.

Unit equal-bound tests capture a Problem, so cannot check SciPy's fixed-coordinate
removal/restoration plus physical unscaling/reporting. One eager SLSQP case under
each backend; no all-fixed problem or method/JIT matrix.

## Defer or reject

- E, feasibility-only native solve: defer. Some distinct native interaction exists,
  but zero-cost derivatives are regression-tested and active native constraints
  are covered by C/current SciPy tests. Add if this becomes a documented workflow;
  never assert a unique optimum for a feasibility problem.
- G, source wavelength->aperture optimization: defer behind this set. Bindings,
  complex propagation/derivatives and Gaussian solves already cover most pieces;
  add if spectral design becomes a first-release workflow or reveals a defect.
- More API-abuse/validation matrices: reject. Existing units and integrations cover
  bad types/shapes/nonfinite values, ambiguous occurrences, external theta/Symbols,
  fixed model edits, dynamic Node rejection, user failures and native infeasibility.
- Generic progressive pipelines: reject duplicates. A/B/F identify specific missing
  handoffs; existing graph/source/composite/result-Symbol loops are substantial.
- Advanced examples: separate migration/release task. Existing CI runner already
  executes examples; migrate old APIs and use it rather than duplicate subprocess
  wrappers or promise convergence for arbitrary nonconvex objectives.
- Stale backend unit fixtures and known Autograd fourth-order failure: existing-test
  migration/bug tasks, not new coverage. No extra reproductions or xfails.
- Mutable binding AST/callable captures: policy unresolved; do not invent deep
  snapshot promises. Multi-input and interior-z probing are not supported APIs.

## Verification

Run affected files under Autograd/JAX, then integration suite. Retain legitimate
new failures, report independent expected/actual behavior and minimal reproducer,
and do not change production. Document exact scenarios/skips. This selection
review changes only this Markdown file and confirms no new production bug.


## Implementation and verification

All selected scenarios implemented, 8 November 2026, with no library changes:

- test_solver_solution.py: a new Solver from Solution.simulation restores the
  first solve's reduced-out clearance constraint after bound relaxation; independent
  physical optimum 0.15 m, with previous solution/defaults intact. A partially fixed
  two-coordinate source solve checks canonical unscaling and optimum angle 0.8.
- test_inverse_design.py: real RayBundle array-residual focus for eager/JIT;
  independent x_out=x_in*(1-L/f), optimum f=L, shapes, angles and reported cost.
- test_backends.py: existing optional native smoke cases now require x+y>=4;
  physical optimum (2.5,1.5), cost 0.5 and actual constraint feasibility checked.
- test_plotting.py: Space/GRIN/Space sampling keeps actual boundary markers and
  only draws free-space interiors, with the boundary-only legend meaning.

Five executable cases added (bundle parametrized eager/JIT), two existing native
cases strengthened. Full integration suite: Autograd **100 passed, 3 skipped**;
JAX **101 passed, 2 skipped**; zero failures or warnings. Optional native tests
skip because NLopt/cyipopt are absent, so their strengthened native expectations
could not be executed locally. Autograd additionally skips the existing JAX branch
rejection case. No production bug was exposed. Deferred/rejected cases above
remain out of scope; no full unit-suite rerun was needed for these test-only edits.
