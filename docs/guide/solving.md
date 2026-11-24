# Solving

A `Solver` finds the values of the variable parameters that best meet your
**objectives**, while satisfying the **constraints**. Mark the parameters you want to
vary, then [build the simulation](building.md). Those parameters form the
[theta vector](building.md#the-variable-vector-theta).
The solver includes the design constraints from the build. Add the goals and
any extra constraints for this solve.

```python
from diffractix import Constraint, GaussianBeam, Plane, Solver, Space, System, ThinLens, units
from diffractix.units import mm, mrad, nm, um

source = GaussianBeam.from_waist(w0=0.75 * mm, wavelength=1064 * nm)
gap = Space(d=60 * mm, label="Gap").variable(d=(40 * mm, 100 * mm))
lens = ThinLens(f=45 * mm, label="Lens").variable(f=(35 * mm, 110 * mm))
working = Space(d=150 * mm - gap.d, label="Working distance")
workpiece = Plane(label="Workpiece")
system = System(gap, lens, working, workpiece, source=source)
system.require(working.d >= 50 * mm)

simulation = system.build()
sim = simulation.symbolic
solver = Solver(simulation)
```

## Objectives

An objective is a quantity to bring close to a target. `target` adds one:

```python
solver.target(value=sim[workpiece].after.w, target=30 * um, label="Spot radius")
solver.target(value=sim[workpiece].after.z0, target=0.0, scale=1 * mm, label="Focus position")
```

Each objective contributes the squared residual `((value - target) / scale)**2`,
multiplied by its `weight` (default 1), to the cost that the optimizer minimizes.

The `scale` sets what counts as a large error. It defaults to the size of the target,
so the first objective above measures a relative error. A target of zero has no size,
so give it a scale explicitly: here, an error of 1 mm in the focus position costs as
much as a spot radius that is off by 100 %. Choose scales so that the residuals you care
about are of order one; the weights then express how important each objective is.

`target(residual)` without `value` minimizes the square of an expression directly. An
`Objective` holds a residual with a weight and a label, and can be passed to `target`
as well.

## Constraints

A constraint must hold at the solution. `require` adds one, as a comparison:

```python
solver.require(sim[lens].after.divergence_angle <= 20 * mrad, label="Paraxial divergence")
solver.require(sim[lens].after.w <= 1 * mm, label="Lens aperture")
```

Constraints added here belong to this optimization only; the requirements of the
design were attached before the build and are included automatically. For a quantity
with both a lower and an upper bound, use `Constraint(quantity, lower_bound=...,
upper_bound=...)`. Bounds on single variables are given when you mark them variable,
not as constraints.

## Quantities as functions

The value of an objective or constraint can also be a function that receives the
result, which is convenient for longer calculations:

```python
def mean_radius(sim):
    return (sim[lens].after.w + sim[workpiece].after.w) / 2

solver.require(Constraint(mean_radius, upper_bound=0.5 * mm, label="Mean radius"))
```

Diffractix first calls the function with the symbolic result, to turn it into an
expression. If the function cannot run on expressions, because it uses an `if` on a
result value, or a function that only accepts numbers, Diffractix instead calls it with
the numerical result each time the optimizer needs its value. It must then return
numbers, and remain differentiable by the backend (see [Backends](backends.md)). With
JAX, a function that uses `if` on a result value also needs `solve(jit=False)`.
CasADi still traces derivatives with `jit=False`, so use `xp.where` for
choices that depend on a result value. Expressions compile with the rest of the model.

## Running the optimization

```python
solution = solver.solve()
print(solution.success, solution.feasible)
print("Gap:", units.format(solution.value(gap.d), mm))
print("Focal length:", units.format(solution.value(lens.f), mm, digits=3))
```

```text title="Output"
True True
Gap: 83.54 mm
Focal length: 66.4 mm
```

`solve()` uses NLopt's SLSQP when NLopt is installed and SciPy's SLSQP otherwise.
Passing a method on its own, such as `method="trust-constr"`, selects SciPy.
Use `optimizer="scipy"`, `optimizer="ipopt"`, or `optimizer="nlopt"` to select an
optimizer explicitly. `options` passes backend-specific settings to the chosen
optimizer. With automatic NLopt selection, it overrides the default settings
`ftol_rel=1e-12`, `xtol_rel=1e-12`, and `maxeval=500`.
`x0`, `x_lower` and `x_upper` override the starting values and bounds for one solve.

The objectives and constraints are fixed at the first `solve()`. Later calls can use
other starting values, bounds or optimizers, but a different problem needs a new
`Solver`.

### Try another starting point

To repeat the same optimization from a different starting point, edit a copy
of the simulation's initial values by parameter and pass it as `x0`. Here the
lens starts at 75 mm and the gap keeps its original starting value.

```python
guess = simulation.initial_values
guess[lens.f] = 75 * mm
retry = solver.solve(x0=guess)
print(retry.success, retry.feasible)
```

```text title="Output"
True True
```

## Reading the solution

A `Solution` reports what the optimizer found:

- `success`: whether the optimizer reported success, and `message`, its reason for
  stopping;
- `feasible`: whether every constraint and bound holds within a tolerance. Check both:
  an optimizer can stop successfully at an infeasible point, and the other way round;
- `solution.value(quantity)`: the solved value of a parameter or any expression of
  parameters, such as `solution.value(working.d)`;
- `solution.run()`: the result of the simulation at the solution;
- `x`, `cost`, `objectives`, `constraints` and `violations`: the solution vector and
  the evaluated objectives and constraints.

`print(solution)` shows all of this as a report. The declared objects are not changed:
`lens.f.value` is still the starting value. `solution.simulation` is a copy of the
Simulation with the solved values as starting values, for a follow-up optimization.

## If the solve fails

- **No variable parameters.** Mark a parameter used by the optical model with
  `variable()`, then [build the simulation again](building.md) and create a new
  `Solver`.
- **Starting values outside the bounds.** Check that each starting value is
  between its lower and upper bounds, including any bounds passed to `solve()`.
  Pass a valid starting point as `x0`. See
  [Try another starting point](#try-another-starting-point).
- **The optimizer returns a solution that misses a constraint.** Read
  `solution.message` for the reason it stopped. Inspect `solution.violations`
  for the constraints and parameter bounds it missed. Check units and limits,
  then [try another starting point](#try-another-starting-point).
