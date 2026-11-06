# Elements and OpticalBase: behavioral specification

## References and ownership

Inspect `core/base.py`, all `elements/*.py`, and the shared-field sections of
`../graph-reference.md` and `../system-simulation-result.md`.
Own `tests/unit/core/` and `tests/unit/elements/`. Other agents own their derived
classes' module tests. Keep shared wrapping coverage here; do not duplicate an
entire cross-module matrix elsewhere.

## Shared OpticalBase contract

```python
@dataclass(eq=False)
class Gain(OpticalBase):
    gain: Node = 2.0
    offset: float = parameter(1.0)
    note: str = "metadata"

obj = Gain()
expression = 3 * obj.gain
handle = obj.gain
obj.gain = 4.0
assert obj.gain is handle  # expression now evaluates to 12
```

OpticalBase is a normal class. It initializes separate per-instance constraint
storage and a label. Discovery includes Node/subclass/optional annotations and
dataclass `parameter()` metadata across inheritance. Ordinary real, complex,
array, boolean and metadata fields are not automatically parameters. Exclude
ClassVar fields. Names preserve declaration order and caches are per concrete
class, including child and sibling classes. Class-cache internals are not public.

Each graph-backed field is an InputNode. Scalars create fixed Parameters with
field name and weak owner association. Existing Nodes retain identity and owner;
expressions remain expressions. None creates an empty handle. Reassignment
updates the existing handle rather than detaching expressions. Invalid scalar
assignment fails without replacing the handle's previous target. Complex and
array Parameters are not supported; explicit expressions may evaluate complex.

`parameters` returns ordered handles; `values` returns current values and None
for unresolved/empty fields. `variable()`/`fixed()` without names select direct
Parameters and skip expressions/empty fields. Named selection rejects derived
or empty fields (TypeError) and unknown names (ValueError). Methods return self.
Constraint attachment is raw and ordered: `require(*constraints)` returns self,
`constraints` is a tuple view, and storage must not leak between instances.
`bindings` defaults to {}; `refractive_index` defaults to None.

## OpticalElement inheritance and lifecycle

OpticalElement is an abstract, keyword-only, eq=False dataclass, inheriting label.
Concrete elements expose matrix and element_length. Custom subclasses adding
dataclass fields should use dataclass(eq=False); __post_init__ overrides should
call super for automatic labels. Explicit labels survive; default labels are
unique by type but absolute counter values are not stable across tests.
System identity rules are tested by the System agent.

`validate()` checks a 2x2 iterable matrix of Nodes or numeric scalars; bools,
strings, arrays as individual entries and malformed dimensions are invalid.
Length must be a Node or real scalar; index may additionally be None. It walks
reachable graph dependencies and rejects cycles, unsupported Nodes and empty
required handles. An empty optional index handle is allowed. Structural
validation is not a blanket build-time numerical-domain validator.

## Built-in physical behavior

Use the physical-angle convention `[x, theta]`, not reduced angle n*theta.
Evaluate matrix entries through current graph API; normalize numerical roots
locally when needed. Independent analytical matrices:

| Element | Matrix / length / medium |
| --- | --- |
| Space(d,n=None) | [[1,d],[0,1]], length d; missing n inherits medium |
| ThinLens(f) | [[1,0],[-1/f,1]], zero length; medium unchanged |
| Mirror(R=inf) | [[1,0],[-2/R,1]], zero length; unfolded propagation |
| Interface(n1,n2,R=inf) | [[1,0],[(n1-n2)/(n2*R),n1/n2]], zero length; output n2 |
| GRIN(d,g,n) | [[cos(gd),d*sinc(gd)],[-g*sin(gd),cos(gd)]], length d |
| ABCD | supplied A,B,C,D; length thickness, optional n |
| GaussianAperture(a,wavelength) | [[1,0],[-i*wavelength/(pi*n*a**2),1]], zero length |
| Plane | identity, zero length, unchanged medium |

GRIN sinc is unnormalized, so g=0 is exactly the Space limit. Preserve finite
first/second derivatives there and existing nonzero physics. Interface planar
radius must yield finite index derivatives, including Hessians; radius may be
positive, negative or infinite. ThinLens f, Mirror R and Interface R reject
exact zero through validate(), not an invented epsilon or !=0 solver constraint.

Built-in constraints: Space d>=0 (+n>0 if explicitly supplied); ABCD
thickness>=0 (+n>0 if supplied); GRIN d>=0,n>0; Interface n1>0,n2>0;
GaussianAperture a>0. ThinLens/Mirror/Plane have no sign constraint on power.
Strict inequalities are Comparison metadata here; normalization belongs elsewhere.
Check they still reference handles after reassignment. Build/optimization applies
these constraints, not every standalone matrix read.

ABCD matrix_val and matrix setter accept 2x2 arrays; reassignment keeps handles
and updates old expressions. Malformed shapes fail. Aperture wavelength defaults
to an independent per-instance Symbol keyed `wavelength`; source supplies its
binding during build. Its public matrix is the unit-index case; System supplies
the local medium through the existing special path.

## Existing coverage and priorities

Existing tests contain independent matrix, curved/planar derivative and GRIN
limit checks worth preserving. Update `requirements` to `constraints`, old
compiler imports to Program APIs, and old discovery-helper fixtures to class
annotations. Add inheritance/ClassVar/cache isolation, failed reassignment,
constraint storage isolation, hidden empty length/matrix dependencies and
validation of shared DAGs (not cycles). Prefer parametrization for sign/shape
boundaries without discarding meaningful analytical examples.

## Ambiguities

- Node-containing annotations are broad (e.g. generic containers); only scalar
  field support is intended. Do not assert containers are supported parameters.
- Arbitrary nonfinite matrix entries are not universally invalid: flat radii
  are allowed. Do not generalize finiteness rules without documentation.
- ABCD documentation says output index; current System enforces continuity for
  an explicit n. Leave that System policy question recorded.
- validate() does not accept bindings; context-dependent singular powers may
  evade exact-zero checks until build. Report concrete failures, do not redesign.
