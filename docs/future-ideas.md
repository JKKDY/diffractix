# Future ideas

Ideas for after 0.0.1. None of these are designed or scheduled; they record intent
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
