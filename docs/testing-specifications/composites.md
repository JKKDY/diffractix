# Composites: behavioral specification

## References, API and ownership

Sources: `composites/composite.py`, Slab, ThickLens, Telescope and FourF;
shared field handling is in `core/base.py`. Read the System working reference.
Own `tests/unit/composites/` only. Do not edit shared-base or System tests.

CompositeElement is a normal OpticalBase subclass and Sequence. Concrete custom
composites can use an ordinary constructor and class-level Node annotations:

```python
class Relay(CompositeElement):
    d: Node
    def __init__(self, d):
        self.d = d
        self.front = ThinLens(f=0.1)
        self.drift = Space(d=self.d)
        self.back = ThinLens(f=0.2)
        super().__init__()
```

The constructor discovers public instance attributes that are elements or
composites, in insertion order, when super().__init__ runs. Private attributes,
ordinary metadata and class-only child attributes are not children. The
resulting composition is a tuple snapshot; parameter handles remain live.

## Views, hierarchy and interactions

- `elements`, `element_names`, `named_elements` provide direct children in order.
- Sequence iteration/len/indexing/slicing match that tuple, including negative
  indices and out-of-range IndexError.
- Nested composites remain direct children; `walk(prefix="")` recursively yields
  `(dotted.path, leaf)` in depth-first propagation order. `leaf_elements` matches
  those leaves. Empty composites are valid collections.
- `element_length` sums leaf lengths, retaining symbolic dependencies. Do not
  multiply or include composite lengths a second time.
- Shared OpticalBase parameter discovery/wrapping also applies to non-dataclass
  composites. Replacing a composite field preserves links in children and
  expressions. Selecting a direct variable does not make dependent expressions
  separate optimization variables.
- `require` attaches raw composite-owned constraints; recursive collection is
  System's responsibility. Child constraints are not copied into the parent's
  own constraints property. Do not impose cross-element dependency restrictions.
- `bindings` combines children recursively. Repeated direct child identity is
  collected once; duplicate keys from distinct providers raise ValueError. Do
  not mutate any child's mapping. Test nested contributors and conflicts.

## Built-in structures and independent physics

Slab(d,n,ambient_n=AMBIENT_N,label=None) contains front Interface, body Space,
back Interface, with child names `front`, `body`, `back`. Its ambient symbol is
resolved at build, not a hidden numeric 1.0. Both boundaries and body share the
composite's stable handles. For ambient na and index n, the aggregate matrix is
[[1,d*na/n],[0,1]], and it returns to na.

ThickLens(d,n,R1=inf,R2=inf,ambient_n=AMBIENT_N,label=None) has the same
front/body/back structure, with curvatures R1/R2 and index transitions. Compare
its aggregate matrix to independently constructed refraction/drift matrices in
the correct multiplication order. Infinite surfaces give the slab limit.

Telescope(f1,f2) contains lens1, space, lens2; spacing f1+f2. For valid nonzero
focal lengths its aggregate C is zero; matrix is
`[[-f2/f1, f1+f2], [0, -f1/f2]]`. Negative focal lengths are not categorically
invalid, but negative total drift violates the attached Space constraint.

FourF(f1,f2) contains input_space, lens1, fourier_space, lens2, output_space;
distances f1, f1+f2, f2. Independent matrix is diagonal with
`A=-f2/f1`, `D=-f1/f2` and B=C=0. Parameter changes must update every linked
child and derived spacing. Preserve physical assertions rather than merely
checking constructor attributes.

## Prior coverage and edge cases

Existing tests cover child order, standard wiring, nested paths and live
reassignment but lack aggregate physical matrix or recursive binding tests.
Keep useful wiring assertions; consolidate repeated constructor boilerplate.
Add empty hierarchy, nested prefix, reused child identity, parent/child parameter
cache independence, per-instance constraints, recursive binding nonmutation,
and repeated conversions after field mutation as appropriate.

Do not instantiate System solely to test traversal. Matrix products may use
the graph interpreter with explicit ambient bindings. Use identity assertions
where symbolic equality would trigger control flow.

## Ambiguities

- A child shared across separate nested composites is not globally deduplicated
  during binding collection, though direct aliases are. Record this distinction
  as unclear policy; do not force one interpretation.
- Cyclic composite containment has no documented cycle error; do not invent a
  supported cyclic hierarchy or encode RecursionError as desired behavior.
- Adding/replacing children after super().__init__ does not refresh contents.
  This is current snapshot behavior, not an explicit mutable-composition API.
