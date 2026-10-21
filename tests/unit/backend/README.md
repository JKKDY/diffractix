# Numerical-backend verification

The parent pytest process may already be using either backend. Tests never reset
backend globals or reconfigure an initialized process.

- `test_configuration.py`: isolated imports, environment/configuration, missing
  optional dependencies, and concurrent first use.
- `test_autograd.py`: explicitly selected Autograd and independent mathematical
  expectations, including identity JIT and eager validation.
- `test_jax.py`: explicitly selected 64-bit JAX, JIT, forward/reverse derivatives,
  nested/fourth derivatives, batching, and validation semantics.
- `test_backend_parity.py`: identical shared calculations under each backend,
  compared with host NumPy at three parameter vectors.
- `tests/integration/test_backend_parity.py`: the actual multi-station example's
  setup, captured before optimization, at three design vectors.

`tests/helpers/backend_cases.py` contains the shared numerical functions.
`tests/helpers/backend.py` launches ordinary Python children, not recursive
pytest sessions. Results are reused within one pytest run. The parent performs
normal pytest assertions, including finite checks and discrepancy diagnostics.

JAX availability is checked with `find_spec`, not by importing it. Missing JAX
skips JAX-specific and parity tests. Missing Autograd fails: it is required.
No optional optimizer is needed for these tests, and no optimization is run.

## Validation contract

`require(condition, message)` rejects false concrete conditions with `ValueError`.
A dynamic condition unresolved during JAX tracing is deliberately **not checked
at runtime**. A false static Python condition can still fail during tracing.
Tests assert both distinctions and ensure the traced guard has no callback-based
assertion. This is not checked-JIT execution; validate concrete values outside
JIT when runtime error reporting is required.

Graph literal construction accepts Python scalars. Tests explicitly convert
static beam properties to Python scalars where needed; active tracers must never
be captured as constants. The real example is tested unchanged, so incompatible
example usage is reported, not silently patched by the test.

## Running

```powershell
$env:DIFFRACTIX_BACKEND = "autograd"
python -m pytest -q
$env:DIFFRACTIX_BACKEND = "jax"  # only when installed
python -m pytest -q
```

The backend-specific tests select their own child processes regardless of the
parent's preference. Comparisons enable JAX float64 explicitly. Basic optical
values use tight tolerances; inverse-length derivatives and higher derivatives
have separately specified absolute tolerances. Native dtype promotion and ragged
array exceptions may differ; the graph tests assert those specific differences
without globally relaxing numerical accuracy or exception checks.
