# Future ideas

Ideas for after 0.1.0. None of these are designed or scheduled; they record intent
so the current architecture can avoid closing doors.

## 1. Multi-beam support and segments

A *segment* is an atomic piece of the system: an incoming beam traverses every
element in it. A segment is a collection of elements and composites.

Segments would let us build several optical paths that diverge and converge, with
multiple inputs running through different parts of the system, combined and split
into different paths.

Open questions: how segments connect (named junctions?), what a combiner or
splitter element is, and how results and symbolic quantities address a stage of a
particular path.

## 2. 2D and 3D layouts

The current one-dimensional view (a single axis, positions by `z`) is probably the
right canonical description of an optical path. Real optical benches span a plane
and are sometimes three-dimensional, so layouts should be supported, including
layout optimization (positions and angles as design variables).

The implementation approach is still open. It likely relates to idea 1 (paths
placed in a plane, folded by mirrors).

## 3. Pluggable simulation methods

Package simulation methods and select one by specification. A method consists of:

- inputs that know how to propagate themselves, and
- elements that carry the information describing how they affect a beam.

`Simulation` then becomes a facilitator of that contract, illustratively:

```python
for element in elements:
    state = state.propagate(element.optical_information)
```

In practice the method has to be named to the simulator, and elements and inputs
may hold optical information for several methods, so the contract will be made
specific per method (ABCD is the first method).

## 4. Hybrid methods

The program picks the simulation method itself where appropriate: ABCD inside the
paraxial approximation, a more expensive method where diffraction matters.
Still tentative; depends on idea 3.

## 5. Better optimization diagnostics

Complex inverse-design problems fail for many reasons: incompatible constraints,
poor initialization, unattainable targets, numerical difficulties. Automatically
reporting the worst constraint violations, the largest objective contributions and
the termination details would help enormously. `Solution` already holds most of the
data (`objectives`, `constraints`, `violations`, `iterations`, `message`), so this is
mostly presentation: for example a `solution.diagnose()` report or an extended
`str(solution)`, not a major API addition.

## 6. Sensitivity and tolerance analysis

Finding an optimum is often only half the problem in optical engineering. A utility
for evaluating a design under parameter perturbations, including Monte Carlo
samples, would be genuinely useful (what happens to the spot size if a distance is
off by 50 um?). It should stay separate from `Solver` rather than complicate
optimization; it only needs `Simulation.run(theta)`, which is already
differentiable and cheap to call repeatedly (JAX could vectorize it).

## 7. Dimension-checked units

`diffractix.core.units` (0.1.0) only scales numbers: `50 * mm` is the float 0.05 and
nothing checks dimensions. A later step could check them without putting unit
objects into the numerics: quantities are converted to SI at the API boundary, and
their dimensions are kept as metadata on Parameters and element parameters (`f` is a
length, `n` is dimensionless). The graph would infer dimensions at compile time, so
`lens.f + space.d` is valid while `lens.f + lens.n` is rejected, and the compiled
program stays plain floats for JAX/Autograd.

Open questions: how plain floats (currently SI by convention) take part, since
existing code such as `ThinLens(f=0.05)` must keep working; and how far inference
reaches into user callables. Quantity objects inside the numerical backends
(Pint/Astropy style) are ruled out because they break JAX and Autograd tracing.

## 8. Gaussian beam factories that follow the ambient index

`GaussianBeam.from_waist`, `from_w_and_R`, `from_divergence` and `from_fiber_tip`
default to `n=1`, while the plain constructor (and `Ray`, `RayBundle`) default to
the system's ambient index. The factories need `n` as a number because the waist
radius fixes `q` only together with the index (`z_R = pi * n * w0**2 / wavelength`).
Following the ambient index would mean keeping the factory inputs (`w0`, waist
position, ...) and computing `q` at build time once `ambient_n` is known, which
changes how `GaussianBeam` stores its state. Kept at `n=1` for 0.1.0, documented,
with the index-mismatch error naming the fix.

## 9. Element-relative symbols

Today a Symbol is bound by a key that is the same for the whole system
(`ambient_n`, the wavelength), and an element can only read its own incoming medium
through the special handling of its refractive index. An element cannot ask for
"the focal length of the element before me" or "the index I am leaving".

Idea: special symbols whose meaning depends on where the element sits, which the
System binds per placement while it resolves the layout. For example
`previous("f")` or `incoming("n")`, resolved at build to the expression of the
preceding element's parameter, or the medium on either side. An element could then
declare relations to its neighbours (an air-spaced doublet matching curvatures, a
coating or aperture sized from the surface before it, a compensator that mirrors
another element) without the user wiring the Parameters together by hand.

This would also give the existing special case a general form: the incoming
refractive index, currently resolved by dedicated code in the build, would be one
such symbol.

A concrete case that exists today: `GaussianAperture._matrix_in_medium(n)`. The
aperture's matrix depends on the index of the medium it sits in, which the element
does not know when it is declared, so the element has a second matrix method that
the build calls with the resolved index (a TODO in the code asks to remove it).
With an element-relative symbol, for instance one generated per element key, the
ordinary `matrix` property would simply use `incoming("n")`, and the build would
bind it like any other symbol. No element would need a special method, and the
build would not need to know which elements have one.

Open questions: what "preceding" means inside composites and for generated gaps
(the leaf before, or the element the user added); how a repeated element gets a
different binding at each placement, since bindings are keyed per system today;
whether to allow reaching further than the direct neighbour (by label or type); and
what happens at the ends of the system and when the neighbour lacks the parameter.
Relates to idea 1 (segments), where "preceding" depends on the path.

## 10. Sparse Jacobians and Hessians

The solver builds dense derivatives: the full constraint Jacobian (`m x n`), the
objective Hessian and the multiplier-weighted constraint Hessian (`n x n` each),
with reverse-mode Jacobians and full Hessians from the backend. That cost grows
quickly with the number of variables `n` and constraints `m`, while the matrices
are often mostly zeros: a constraint on one lens's thickness depends on one
variable, an aperture limit only on the parameters upstream of that element.
`solver/problem.py` already carries a TODO for this.

Directions, roughly in order of effort:

- Know the sparsity pattern for free. The compiled graph programs record which
  theta entries each output depends on, so the structural pattern of the Jacobian
  (and of the Hessian, from pairs that meet in a nonlinear operation) can be read
  off the program instead of being detected numerically. Runtime result quantities
  depend on everything upstream of their stage, which gives a block-triangular
  pattern along the optical path.
- Hand the pattern to optimizers that use it. IPOPT takes the Jacobian and Hessian
  structure and only the nonzero values; SciPy's trust-constr accepts sparse
  matrices and linear operators. SLSQP is dense by design and would stay as it is.
- Compute fewer derivative passes. With the pattern known, graph colouring lets
  one forward or reverse pass recover several columns or rows at once (compressed
  Jacobians), instead of one pass per variable.
- Avoid forming Hessians at all where possible: Hessian-vector products
  (a forward pass over the gradient) for methods that only need products, or
  quasi-Newton approximations (IPOPT's limited-memory mode) for large problems.

Open questions: where the dense-to-sparse switch should happen automatically (small
problems are faster dense); how sparse results pass through the variable and
constraint scaling and the constraint reduction, which assume dense arrays today;
and how the pattern survives JIT, since JAX needs static shapes for the nonzeros.
Relates to idea 1 (segments): several independent paths make the matrices block
structured, which is where sparsity pays most.

## 11. Objective norms

Every objective adds `weight * sum(residual**2)` to the cost, a squared L2 norm.
Let the user choose the norm per objective, for example L1 (robust to outliers,
favours exact zeros) or L-infinity (worst case, such as the largest beam radius
along a system). Both are non-smooth, so they need care with gradient-based
optimizers: a smooth approximation, or the usual reformulation with auxiliary
variables and constraints (`|r| <= t`, minimize `t`). Open question: whether the
norm belongs on `Objective` or on `Solver.target`.

## 12. Complex parameters

Complex quantities (a Gaussian beam's `q`, complex ABCD entries such as an aperture's)
can only be constants today. A complex parameter would let the optimizer vary them.
Design agreed with the maintainer: back it by two real `Parameter`s, so theta gets
a real entry for the real part and one for the imaginary part, and in the graph it is
just `re + 1j * im`. The optimizers (SciPy, IPOPT, NLopt) stay purely real, and
gradients stay ordinary real gradients. Needs: a way to create one (a factory or a
complex flavour of `Parameter`), bounds per part, `variable()`/`fixed()` acting on
both, `parameter_data` and `Solution.value()` reporting the pair as one complex value,
printing, and fields such as `GaussianBeam.q` accepting it. Do not add special-case
complex fields to elements or states for this; it belongs in the graph.

## 13. Elliptical Gaussian beams and per-axis elements

A beam with different beam parameters in x and y (`q_x`, `q_y`), such as a diode
laser, and elements that act differently on the two axes, such as cylindrical lenses,
anamorphic prism pairs and mirrors at an angle. Circularizing a diode beam is the
typical design problem.

Two parts, which are only useful together:

- An elliptical beam state. Through round optics each axis propagates independently
  with the same ABCD matrix, so the simulation is unchanged: the state propagates
  `q_x` and `q_y` with the matrix it is given. The work is in the state: per-axis
  properties (`w_x`, `w_y`, `R_x`, ...), the symbolic result, result columns,
  plotting both axes and factories.
- Elements with a matrix per axis, for example an optional `matrix_x` and `matrix_y`
  that default to `matrix`. This changes the element contract: the compile step keeps
  two matrices per stage and `propagate` takes both.

Open questions: whether `Ray` and `RayBundle` gain a y coordinate or stay x only, and
how plots show two axes.

Out of scope: general astigmatism, where a rotated cylindrical lens couples x and y.
It needs 4x4 ray matrices and a complex 2x2 beam matrix, a different propagation
model, so it belongs with idea 3 (pluggable simulation methods).

## 14. Solving several simulations together

Let one `Solver` optimize several Simulations at once, coupled by shared Parameters
and shared elements: for example the same optics at several wavelengths, a system
used with two different sources, or one design checked at several focus positions.
Objectives and constraints can then combine quantities from different simulations,
such as the difference between two beams' focus positions.

The variables would be the union of the simulations' variable Parameters, matched by
identity, so a shared Parameter has one entry in the combined theta. Each
simulation's programs are remapped into that order, as objectives and constraints are
today. Each simulation keeps its own symbolic result, and `Solution` reports values
and runs per simulation.

This is likely a prerequisite for idea 1 (multi-beam setups and segments): several
paths through shared elements are several coupled simulations.

Open questions: how a user addresses a simulation's result in an objective (one
symbolic result per simulation, or one combined object); how `solution.run()` and
`solution.simulation` look for several simulations; and how a shared element that
appears in several systems with different positions is handled.

## 15. Watching an optimization: an iteration callback

Let `Solver.solve()` take one optional callback, for example `on_iteration`, that is
called after each optimizer iteration with a copy of the current design vector `theta`
in physical SI units. The user can record it, run `simulation.run(theta)` on it, or
update a plot or UI. This makes it possible to replay how an optical system converges:
lenses moving, waists changing, constraints becoming active. It also shows parameter
coupling and why a solve stalls, which complements idea 5 (diagnostics).

`Simulation.run(theta)` already accepts any parameter vector, so no change to the
graph compiler or the simulation is needed. The change is in the optimizer layer:

- SciPy: `minimize` takes a `callback` for both SLSQP and trust-constr. It receives
  the scaled variables `y`, which must be converted back with `x = x_center +
  x_scale * y` (`Scaling.unscale`) before the user's callback sees them.
- IPOPT: the adapter already has an `intermediate` callback that counts iterations,
  so it is the natural hook. Whether the current iterate can be read there depends on
  the CyIpopt version; check before relying on it.
- NLopt: only objective evaluations are visible, with no accepted-iteration callback.
  Calling the user's function there would mix trial points with real iterations, so
  either skip NLopt or say plainly what the callback receives.

Keep it to the single callback. No generators, asynchronous solver classes or event
architecture. The callback does not report the starting and final designs, so a
history that wants both adds them itself (`x0` and `solution.x`).

Open questions: the exact name and signature (only `theta`, or also the iteration
number and cost); what happens when the callback raises (stop the solve, or let the
error out); and how to say that intermediate designs can violate constraints and may
give non-finite results, so a UI should treat them as candidates, not solutions.

A browser UI could first record the vectors and replay them at a chosen speed, since
many optical solves finish in 5 to 20 iterations, too fast to watch live.
