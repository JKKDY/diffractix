# Optical inputs: behavioral specification

## References and ownership

Read `input/input.py`, `paraxial_input.py`, Ray, RayBundle, GaussianBeam,
`core/base.py`, `system/symbols.py`, and the two current working references.
Own `tests/unit/input/` only. Preserve existing independent ray/bundle physics
and factory derivatives. Do not edit System or propagation integration tests.

## Declarative versus numerical lifecycle

OpticalInput is a keyword-only eq=False dataclass inheriting OpticalBase.
Fields are graph-backed only through Node/subclass annotations or parameter()
metadata. Ordinary float, bool, array, complex and metadata fields remain
ordinary. This is the same wrapping contract as elements, not auto-real wrapping.
ParaxialState adds an abstract propagate(A,B,C,D,n) and result_columns=().

```python
ray = Ray(x=1e-3, theta=0.01, n=1.0)
ray.theta.variable()
numerical = ray.numeric()
out = numerical.propagate(1.0, 0.1, 0.0, 1.0, n=1.0)
# out.x == 2e-3; ray remains declarative
```

`numeric(bindings={})` evaluates constructor graph fields at current Parameter
values, even when variable. Explicit bindings override Symbol defaults, may
contain Node expressions, and are not modified. Missing required Symbols raise
KeyError; cycles and empty required dependencies use graph errors. Optional
direct empty fields become None. `bindings=None` is not currently normalized;
the signature promises a mapping, not explicit None.

Numerical objects use a cached generated subclass of the concrete input type.
It retains inherited properties, constructors and __post_init__, but bypasses
wrapping for construction and later assignment; parameter_names/parameters are
empty. It accepts backend scalars and tracers without Python float conversion.
Repeated numeric conversion returns a new instance of the same numerical type.
Different input classes get different types. Calling numerical_type on the
generated type returns that type.

Constructor init fields include ordinary numerical fields. init=False fields
are recomputed by __post_init__, not copied/evaluated from stale declarative
derived fields. Custom constructors must accept init fields as keywords. A
no-graph-field input still supports numeric(). Ordinary arrays are preserved
by reference here; numeric is not a deep-copy API. System owns build snapshots.
Builtin propagation uses type(self), so numerical types remain numerical.

## Built-in APIs and independent numerical expectations

| Type | Graph fields | Ordinary fields | Main numerical behavior |
| --- | --- | --- | --- |
| Ray | x,theta,n | label | vector=[x,theta]; x_at_z=x+z*theta |
| RayBundle | n | x,theta arrays,label | vectors shape (2,N); per-ray propagation |
| GaussianBeam | wavelength,n | complex q,label | q'=(A*q+B)/(C*q+D) |

Default n is a fresh Symbol with ambient_n key; AMBIENT_N currently has no
default. Standalone numerical conversion must supply n or an ambient binding.
Do not assume `.numeric()` silently chooses air. RayBundle factories and
Gaussian factories explicitly default n=1.0, unlike direct constructors.

Ray and RayBundle follow physical angle theta: free-space x'=x+d*theta,
thin-lens theta'=theta-x/f, planar refraction theta'=n1*theta/n2, curved
refraction includes (n1-n2)*x/(n2*R). No additional division by n in free space.
Each propagation creates a new state and tracks passed outgoing n. Sequential
propagation agrees with ordered matrix multiplication and leaves source intact.
Compare bundle output against independent Ray instances, not only broadcasting.

RayBundle validates matching 1D x/theta shapes. Empty and one-ray arrays are
supported. It exposes size, vectors, x_at_z, centroid, mean_angle,
position_spread, angular_spread and rms_x. Spreads use population standard
deviation; rms_x is about the axis, not centroid. parallel samples [-radius,radius]
with zero angles; converging uses theta=-x/focal_distance. Avoid unspecified
factory invalid-domain policy and derivatives at zero spread.

Gaussian mathematical properties use `1/q=1/R-i*lambda/(pi*n*w**2)`,
z_r=Im(q), z0=-Re(q), w0=sqrt(z_r*lambda/(pi*n)), divergence=lambda/(pi*n*w0),
NA=lambda/(pi*w0), gouy=atan2(Re(q),Im(q)), b=2*z_r. R is infinite at the waist
and its inactive division must not poison derivatives. Free space gives q+d.
`bindings` contributes wavelength keyed by WAVELENGTH. Public direct numerical
properties should be tested on numerical objects, not InputNode fields.

Factories: from_waist, from_w_and_R, from_divergence and from_fiber_tip. Widths,
wavelength,n and positive mode descriptors require real finite positive concrete
inputs (not bool). Waist location finite; R nonzero/non-NaN, either sign/infinity.
NA takes precedence over MFD; at least one must be supplied. Numerical classes
allow backend array factory arguments and broadcasting. Public graph fields
are real scalar only. Factories need not make complex q optimizable.

overlap_with checks compatible media (absolute tolerance 1e-6) and wavelengths
(1e-9), then returns symmetric mode overlap; identical beams yield 1. Independently
check mismatched waist radii and propagation invariance in compatible lossless
systems. GaussianBeam construction and w_R_from_q reject nonfinite q or a nonpositive
imaginary part with ValueError, before division. No epsilon threshold is used; traced validation follows
backend.require's documented unchecked-runtime policy.

## Coverage and limits

Cover Node-valued bindings, inherited/init=False dataclass fields, cached numerical
subclasses, per-instance constraints and symbols, numerical propagation, analytical
Gaussian properties/overlap and representative nested differentiation/JIT.
Shared wrapping behavior belongs in OpticalBase tests rather than repeated matrices.

Ordinary complex/array fields remain numerical; public scalar graph fields are
not arrays. Custom propagation that hardcodes a declarative constructor may lose
its numerical subclass. Empty bundle construction and propagation are supported;
factory-domain and empty-statistics validation is not yet a settled contract.
See ../qa-status.md for the remaining policies.
