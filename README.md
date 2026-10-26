# Diffractix

A differentiable gaussian optics simulation!

The goal is to solve inverse-design problems involving beams in the Gaussian paraxial approximation.

Current features:
- Gaussian beam representation using complex q-parameters
- ABCD propagation through spaces, lenses, mirrors, interfaces, apertures, and thick-lens/slab composites
- Declarative optical system builder with relative and absolute positioning
- Differentiable parameter graph for optimization with autograd
- Layout and refractive-index consistency checks
- Unit tests for beam physics, optical elements, system compilation, and graph behavior

Status:
This is still very much a work in progress. Forward simulation and differentiable system compilation work; the inverse-design API is still under development.

### Plotting

Install the optional renderer with `python -m pip install 'diffractix[plot]'`:

```python
from diffractix import GaussianBeam, Space, System, ThinLens

system = System(Space(d=0.06), ThinLens(f=0.08), Space(d=0.18),
                input=GaussianBeam.from_waist(0.3e-3, 1064e-9))
simulation = system.build()
result = simulation.run()
fig, ax = result.plot()  # Or: from diffractix.plotting import plot; plot(result)
```

`result.plot(ax=ax, show_elements=True, show_axis=True)` also accepts existing
Matplotlib axes and never shows or saves the figure automatically. Plotting
assumes input lengths in metres; z is labelled in m and transverse values in mm.

This is a schematic 2D paraxial visualization for Ray, RayBundle, GaussianBeam,
and their subclasses. Continuous curves are sampled using the entrance state
of each Space, including Spaces inside composites. Other thick elements
(such as GRIN or black-box ABCD components) show only their extents and boundary
states; no internal trajectory is invented. Sampling recognizes `Space` by the
existing element type name; a custom class
literally named `Space` is therefore assumed to have homogeneous-space behavior.
Bundles show up to 41 evenly spaced rays.
Element markers are schematic, not mechanical drawings. Use concrete evaluated
results outside differentiation/JIT. See [the plotting example](examples/plotting.py)
for Ray, RayBundle, and GaussianBeam figures.
