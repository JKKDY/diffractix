# Test organization

- `unit/`: focused graph, element, composite, system-pass, simulation, and solver contracts. Numerical dependencies are allowed; mocked backend adapter tests run even without optional optimizers.
- `integration/`: real build/propagation pipelines, solver tracing and runtime evaluation, backend optimization, and optical inverse-design workflows.
- `e2e/`: complete design workflows through the public API only: declare a system with unit constants, build, optimize, read the report and plot. Each is checked against physics worked out independently (hand geometric optics or plain NumPy matrices), and the failure workflows (an infeasible requirement, a stopped optimization continued from its solution) are covered too.
- `helpers/`: small shared solver snapshots and analytical optimization problems. Other fixtures stay local to their test modules.

Shared optical parameter contracts live in `unit/core/`. `unit/input/` contains
declarative input conversion and independent Ray, RayBundle, and GaussianBeam
physics tests. `integration/test_input_propagation.py` exercises the complete
graph-to-numerical-state boundary, including differentiation through bindings.

Numerical backend configuration and explicitly selected Autograd/JAX behavior
live in `unit/backend/`. See its [README](unit/backend/README.md)
for process isolation, optional JAX handling, and eager-versus-traced validation.
Comparisons spanning real optical systems and solver callbacks live in
`integration/test_numerical_parity.py`, using small self-contained cases from
`helpers/backend_cases.py` with independent analytical expectations.

Happy paths, invalid inputs, and regression cases stay beside the feature they exercise. There is no separate edge-case or regression suite.

Which tier a test belongs in:

- one class or function, no optimizer run: `unit/`;
- several parts working together on a small case (a build, a propagation, one solve): `integration/`;
- a whole user task from declaration to report: `e2e/`.

Cases that differ only in their inputs share one parametrized test (for example the
zero focal length and zero radius checks in `unit/elements/test_elements.py`). A
behaviour is asserted in one place: the JIT error hint lives in
`unit/solver/test_solve_validation.py`, parameter bounds in
`unit/graph/test_construction.py`.

Run from the repository root with the development environment installed:

```console
pytest
pytest tests/unit
pytest tests/integration
pytest tests/e2e
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

Standalone numerical propagation fixtures use `numeric()` or `numerical_type()`;
`System` itself only accepts declarative inputs. Examples are not executed by
pytest; `scripts/run_examples.py` runs them separately.

Real IPOPT tests use `pytest.importorskip("cyipopt")`; real NLopt tests use `pytest.importorskip("nlopt")`. Missing optional packages skip only their real optimizer tests, not mocked adapters or the rest of the integration suite. Optimizer markers select real optimizer tests; unit adapter tests are selected by directory.

Pytest discovers only `tests/`, excluding archived suites. Importlib mode permits matching filenames in the unit and integration directories.
