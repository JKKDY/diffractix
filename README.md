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
