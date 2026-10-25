# Test organization

- `unit/`: focused graph, element, composite, system-pass, simulation, and solver contracts. Numerical dependencies are allowed; mocked backend adapter tests run even without optional optimizers.
- `integration/`: real build/propagation pipelines, solver tracing and runtime evaluation, backend optimization, and optical inverse-design workflows.
- `helpers/`: small shared solver snapshots and analytical optimization problems. Other fixtures stay local to their test modules.

Shared optical parameter contracts live in `unit/core/`. `unit/input/` contains
declarative input conversion and independent Ray, RayBundle, and GaussianBeam
physics tests. `integration/test_input_propagation.py` exercises the complete
graph-to-numerical-state boundary, including differentiation through bindings.

Numerical backend configuration and explicitly selected Autograd/JAX behavior
live in `unit/backend/`. See its [README](unit/backend/README.md)
for process isolation, optional JAX handling, and eager-versus-traced validation.
Comparisons spanning real optical systems and solver callbacks live in
`integration/test_numerical_parity.py`.
The real multi-station parity scenario lives in `integration/test_backend_parity.py`;
it reuses the example without running optimization.

Happy paths, invalid inputs, and regression cases stay beside the feature they exercise. There is no separate edge-case or regression suite.

Run from the repository root with the development environment installed:

```console
pytest
pytest tests/unit
pytest tests/integration
pytest -m scipy
pytest -m "ipopt or nlopt"
```

To verify shared tests with each installed numerical backend, select the backend
before starting pytest:

```powershell
$env:DIFFRACTIX_BACKEND = "autograd"
python -m pytest tests
$env:DIFFRACTIX_BACKEND = "jax"
python -m pytest tests
```

Numerical propagation fixtures explicitly use `numeric()` or `numerical_type()`;
this does not stand in for future symbolic-source compilation into System.
The advanced example is executed unchanged, so incompatible example usage remains
a visible failure rather than being skipped or patched by a test helper.

Real IPOPT tests use `pytest.importorskip("cyipopt")`; real NLopt tests use `pytest.importorskip("nlopt")`. Missing optional packages skip only their real backend tests, not mocked adapters or the rest of the integration suite. Backend markers select real optimizer tests; unit adapter tests are selected by directory.

Pytest discovers only `tests/`, excluding archived suites. Importlib mode permits matching filenames in the unit and integration directories.
