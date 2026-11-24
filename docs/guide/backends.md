# Backends

Diffractix computes derivatives by automatic differentiation, with one of three
numerical backends:

- **Autograd** is included in the base install. It differentiates eager numerical
  evaluation and treats `jit` as an identity operation.
- **JAX** is available with `pip install "diffractix[jax]"`. It supports just-in-time
  compilation (JIT), which can speed up repeated numerical evaluations.
- **CasADi** is available with `pip install "diffractix[casadi]"`. It traces
  numerical functions with SX symbols and compiles their values and derivatives
  for repeated evaluation.

The examples in this documentation run on all three backends. Use the
[backend's numerical functions](#writing-numerical-code) in custom code
that needs to run across backends.

## Choosing a backend

By default Diffractix uses JAX if it is installed, and Autograd otherwise. Set
`DIFFRACTIX_BACKEND` to `jax`, `autograd`, `casadi`, or `auto` before starting Python.
To choose from Python, import `backend` from `diffractix` and call
`backend.configure("autograd")` before the first numerical calculation. The
call takes precedence over the environment variable.

The first numerical calculation fixes the backend for that Python process.
`backend.configure()` raises an error after that point. To check which backend
is active, call `backend.current_backend()`. It returns `None` before the first
numerical calculation.

Diffractix enables double precision for JAX by default. Set `JAX_ENABLE_X64`
before the first numerical calculation to choose the precision setting explicitly.

## Derivatives of your own functions

Use `grad`, `jacobian`, and `hessian` from `diffractix.backend` to differentiate
functions that use operations supported by the active backend. This example
uses the selected backend to find how the ray height changes with each
variable in [the design vector `theta`](building.md#the-variable-vector-theta).

```python
from diffractix import Plane, Ray, Space, System, ThinLens
from diffractix.backend import grad
from diffractix.units import mm

gap = Space(d=50 * mm).variable(d=(10 * mm, 100 * mm))
lens = ThinLens(f=100 * mm).variable(f=(50 * mm, 200 * mm))
image = Plane(label="Image")
simulation = System(gap, lens, Space(d=100 * mm), image,
                    source=Ray(x=1 * mm, theta=0.0)).build()

def height(theta):
    return simulation.run(theta)[image].after.x

print(grad(height)(simulation.initial_values.values))
```

```text title="Output"
[0.   0.01]
```

## Writing numerical code

Objectives and constraints written as [expressions](parameters.md#expressions) need
no care: Diffractix compiles them for the backend. Code that runs on numbers, such as
a function of the result that cannot be turned into an expression, a custom element,
or a function you differentiate yourself, must be differentiable by the backend:

- use `diffractix.backend.xp` for array functions (`xp.exp`, `xp.where`, ...). It is
  `jax.numpy`, `autograd.numpy`, or the CasADi NumPy interface;
- avoid in-place array changes. During differentiation, keep values in the
  backend rather than converting them with `float()` or `np.asarray()`;
- with JAX and JIT, an `if` cannot depend on a computed value. Use `xp.where`, or
  solve with `solve(jit=False)`.

The [backend API page](../api/backend.md) documents `xp`, the derivative functions
and `jit`.

## CasADi tracing

Pass positional numerical inputs with fixed shapes to a transformed function.
Scalars, vectors and numerical arrays keep their shapes at the function boundary.
SX arithmetic uses double precision and rejects integer inputs outside its exact
integer range.
`grad` and `hessian` require a real scalar result. `jacobian` accepts a real
scalar or array result. Derivatives use one real floating-point input selected by
`argnum`. `jit` also accepts complex numerical results and returns ordinary complex
values. Use `static_argnums` for hashable arguments that should stay as Python
values during tracing.

Traced values work with these `xp` operations.

- Arithmetic, comparisons, Boolean logic, `abs`, `square`, `sqrt`, `exp`, `log`,
  `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arctan2`, `minimum`,
  `maximum`, `isfinite`, `isinf`, and `isnan`.
- `array`, `asarray`, `stack`, `concatenate`, `ravel`, `reshape`, `full`,
  `linspace` with scalar endpoints, `sum`, `all`, `mean`, `std`, `dot` on
  vectors and matrices, and `where`. The reductions operate over the whole array.
- Addition, subtraction, multiplication, division, negation, squaring, `abs`,
  `sin`, `real`, `imag`, and `conj` for complex traced values. These cover
  Gaussian propagation and observables, including complex ABCD coefficients.

Use concrete indices and write choices that depend on traced values with
`xp.where`. Python branching, conversion to a host scalar or array, and
unsupported traced operations raise a tracing error. A transformed function can
read a `SimulationResult` during tracing and must return numerical data.

The solver first tries to build symbolic expressions from callables and defers
those that only work on numerical results. CasADi still traces any deferred
callables it needs to differentiate, including with `solve(jit=False)`. Write
both branches of `xp.where` so they are safe to evaluate.
