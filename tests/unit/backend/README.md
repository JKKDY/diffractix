# Numerical-backend verification

Backend-specific tests use separate Python processes. They never reconfigure an
initialized numerical backend. Shared numerical tests use diffractix.backend.

- test_configuration.py: selection, imports, missing optional dependencies and
  concurrent first use.
- test_autograd.py: identity JIT, analytical complex gradients/Hessians,
  forward/reverse Jacobians and concrete validation.
- test_jax.py: 64-bit precision, JIT/static arguments, Jacobians, Hessians,
  batching, literal/tracer handling and concrete versus traced validation.
- tests/integration/test_numerical_parity.py: small graph, Gaussian propagation
  and actual inverse-design calculations compared across numerical backends.

Current Solver algorithms require first and second derivatives. Explicit
fourth-order tests are outside the release contract; representative nested
transformation tests remain in graph/input/integration suites. This does not
promise general fourth-order support or alter the numerical implementation.

The parity helpers launch ordinary Python children, cache completed results per
run and compare finite values against independent analytical expectations. JAX
checks skip when the optional package is absent; Autograd is required.

require rejects false concrete conditions. JAX predicates depending on tracers
are deliberately unchecked at runtime; static false conditions still raise.
Graph literals can capture concrete scalars/arrays, but cannot capture active
tracers as constants. Numerical differentiation must remain in its active graph.

Run pytest tests with DIFFRACTIX_BACKEND=autograd or jax in separate processes.
