# Backends

Diffractix computes derivatives by automatic differentiation, with one of two
numerical backends:

- **Autograd**, always installed. It runs on any platform where NumPy does.
- **JAX**, installed with `pip install "diffractix[jax]"`. It compiles the model
  just in time (JIT), which makes repeated evaluation faster, especially for larger
  problems.

Your code is the same with either backend. The examples in this documentation run on
both.

## Choosing a backend

By default Diffractix uses JAX if it is installed, and Autograd otherwise. To choose
explicitly, set the environment variable `DIFFRACTIX_BACKEND` to `jax`, `autograd` or
`auto`, or call `configure` before the first calculation:

```python
from diffractix import backend

backend.configure("autograd")
```

The choice is made once per Python process, at the first numerical calculation, and
cannot be changed afterwards; `configure` then raises an error. `backend.current_backend()`
returns the backend in use, or `None` before the first calculation.

With JAX, Diffractix enables double precision, so results agree with Autograd.

## Derivatives of your own functions

Everything a run computes is differentiable with respect to `theta`. The backend's
derivative functions, `grad`, `jacobian` and `hessian` in `diffractix.backend`, work on
any function built from it:

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
  `jax.numpy` or `autograd.numpy`, depending on the backend;
- do not modify arrays in place, and do not convert values to Python numbers with
  `float()` or `np.asarray()`;
- with JAX and JIT, an `if` cannot depend on a computed value. Use `xp.where`, or
  solve with `solve(jit=False)`.

The [backend API page](../api/backend.md) documents `xp`, the derivative functions
and `jit`.
