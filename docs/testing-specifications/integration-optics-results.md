# Optical integration test review

8 November 2026. Source and dependency/configuration files were not changed.
Owned files: test_system_build.py, test_simulation.py,
test_source_compilation.py, test_sequence_propagation.py,
test_element_propagation.py, test_ray.py, test_ray_bundle.py,
test_input_propagation.py in tests/integration.

## What was tested

- Public System -> Program -> Simulation -> Result propagation with real Ray,
  RayBundle, GaussianBeam, a custom dataclass ParaxialState and custom optical
  translation. Numerical subclasses retain their state type during propagation.
- Independent physical-angle ABCD formulas for positive/negative focal lengths,
  planar/positive/negative curved interfaces, Slab and ThickLens. Refractive
  indices change at surfaces; longitudinal positions remain boundary samples.
- Bundles match separately compiled individual rays through seven element/path
  combinations, including material transitions and composite chains. Empty and
  single-ray bundles retain vector shapes.
- Source parameters and element expressions share canonical theta. Numerical
  gradients, Jacobians, Hessians and JIT results agree with independent analytic
  expressions; repeated evaluation preserves declarations.
- Planar-interface index derivatives remain finite. GRIN zero gradient matches
  the free-space limit; a nonzero gradient matches the trigonometric matrix.
- Gaussian apertures use source wavelength bindings and each placement's local
  medium, including reuse of one aperture across an interface. Beam-width values
  and derivatives agree with the inverse-square-width envelope formula.
- Native Program outputs retain real wavelength/index alongside complex matrix
  coefficients; float32 bundle arithmetic follows native scalar promotion rather
  than the removed packed-array coercion path.
- Nested composite paths, shared design parameters, generated absolute-placement
  gaps and parameter-output inspection metadata match evaluated optical values.
  Composite/child/system constraints keep order and are not duplicated when the
  same composite is reused or rebuilt.
- Fixed negative generated gaps fail as unsatisfiable; variable negative gaps
  remain optimizable constraints and become satisfied at a feasible theta.
- Original source array views, q/wavelength changes and element field reassignment
  cannot change existing simulations/results. Independent InitialValues edits do
  not mutate defaults.
- Sad paths: missing optical input, unresolved Symbols, external requirement
  variables, fixed medium mismatch, unsatisfiable system/element constraints,
  wrong theta lengths, ambiguous/missing/repeated element lookup and invalid
  occurrence type/range.
- Existing standalone OpticalInput conversion/propagation checks were retained,
  including numerical custom subclass behavior, complex fields and nested AD.

## Consolidation

Replaced obsolete large suites targeting compile_ast/CompiledAST, simulation.graph,
source/steps/parameter_graph/element_info, numerical source System inputs,
result.at()/after()/stacked properties and old constraint naming with focused
current-interface scenarios. Independent optical equations, material transition,
aperture, shared parameter, repeated-instance, snapshot and derivative checks were
preserved. Tests duplicating base-class/unit validation were not mechanically
ported. Actual solver convergence was removed from the ray suite because inverse
 design tests are owned by the solver workstream.

## Verification

All eight owned files were run together in separate existing-venv processes:

- Autograd: **50 passed**, no failures/skips/warnings, 0.92 seconds.
- JAX: **50 passed**, no failures/skips/warnings, 12.00 seconds.

Command: set DIFFRACTIX_BACKEND to autograd or jax, then run
`.venv/Scripts/python.exe -m pytest <eight owned files> -q --tb=short -p no:cacheprovider`.

Initial setup assumptions corrected during test authoring: graph-backed ABCD
complex coefficients require explicit Literal (ordinary graph fields accept real
scalars/Nodes), and native float32 arithmetic is compared to a Python scalar
rather than a forced np.float64 scalar. Neither was an implementation defect.
No production bugs were confirmed. No xfail or behavior-suppressing skip added.

## Deliberate gaps

General interior-z probing is not supported by the current public API. Deferred
callable closure mutation and binding-AST snapshot policy remain outside this
snapshot contract. Fourth-order Autograd stacking is previously deferred; tests
cover representative nested/second-order transformations here. Native optimizer
convergence and complete Solution reporting belong to the solver workstream.
