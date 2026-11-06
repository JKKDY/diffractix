# Graphs: behavioral specification

## References and scope

Sources: `graph/node.py`, `graph_utils.py`, `compile.py`, `program.py`,
`program_utils.py`, `ops.py`, `relations.py`, `errors.py`, and public exports.
Primary documentation: `../graph-reference.md`. It still calls the parameter
mapping `variables`; source has renamed it to `parameter_indices`.
Own `tests/unit/graph/` only. Include the Node base, interpreter, traversal,
program execution, optimization, fusion and remapping.

## Public API and usage

```python
x = Parameter(2.0, name="distance").variable(bounds=(0.0, 4.0))
root = x**2 + Symbol(("result", "width"))
program = compile_graph((root, x))
values = program.evaluate([3.0], {("result", "width"): 0.5})
# (9.5, 3.0), in root order
root.evaluate({("result", "width"): 0.5})  # uses current x.value
```

Documented node contracts:

- Literal embeds a numeric constant with a read-only public `value` property.
- Parameter is mutable, real scalar, fixed by default. `bound`, `variable`, and
  `fixed` return self. A variable leaf consumes one theta entry by identity.
- `variable(bounds=(lo, hi))` and individual endpoints share validation. None
  leaves an endpoint unchanged; infinities explicitly unbind it. Non-None
  competing endpoint arguments conflict. Wrong scalar types and bool are invalid;
  NaN bounds, inverted bounds and exclusion of the current value are invalid.
- Symbol keys are hashable. Distinct Symbols with equal keys share a runtime
  slot. None means no default. `set_default` is fluent. `.value` cannot resolve it.
- InputNode is a stable mutable handle. Attribute reads, writes and parameter
  methods forward to its target. Replacing `.node` affects existing ASTs.
- Arithmetic and unary methods build Nodes. Comparisons build Comparison objects,
  not boolean values or arithmetic Nodes. Both Node and Comparison truth testing
  raise SymbolicControlFlowError, a TypeError subtype. Unrelated TypeErrors stay
  ordinary TypeErrors. Nodes are unhashable because equality is symbolic.

## Interpreter, traversal and variability

`evaluate_ast(root, bindings={}, parameter_values=None)` and `root.evaluate`
calculate one root. Explicit Symbol bindings precede defaults and may themselves
be Nodes, recursively resolved. Missing Symbols raise KeyError containing the key.
Without `parameter_values`, fixed and variable Parameters both use current values.
With it, every encountered Parameter requires an `id(parameter)` entry; missing
entries fail instead of falling back. Preserve backend values and differentiation.

`walk_ast` yields roots before descendants, left before right, once per identity;
it follows explicit Node-valued bindings, not Symbol defaults. Shared DAGs are
not cycles. Active-path cycles raise ASTCycleError. Empty handles raise
UnresolvedInputError. Unsupported node subclasses raise UnsupportedNodeError.
Non-Node roots are TypeErrors. Empty roots yield no nodes.

`has_variable_parameters(bindings)` uses all reachable dependencies and requires
every Symbol to be explicitly bound, even when it has a default or a different
branch already contains a variable. `.is_variable` instead follows the raw graph
and reports False for Symbols. Do not conflate these methods.

## Compilation lifecycle and invariants

`precompile(roots, bindings={})` snapshots AST structure and fixed values in a
Program. Variable indices follow deterministic first occurrence. Repeated
Parameters share one index; equal-valued distinct Parameters do not. Mutating
the AST, a handle, fixed Parameter, or scalar binding afterward must not change
the program. Changing variable flags afterward must not change compiled inputs.
The new Program does not own initial values or accept `parameter_snapshot`.

Bound Symbols resolve recursively and leave no runtime entry. Unbound Symbols
remain runtime inputs, including Symbols with defaults. Runtime values can be
real, complex or arrays. `Program.evaluate` returns a tuple of native roots,
without common dtype promotion or stacking. Empty programs return (). Direct
constant/input and duplicate outputs preserve order. Wrong theta length raises
ValueError; missing runtime keys raise KeyError. Extra bindings are harmless.

`compile_graph` optimizes precompilation. Optimization must preserve values,
derivatives, signed zero and operand order; it must not use algebraic identities
such as x*0 or commute inputs. It conservatively folds constants, shares
equivalent computations, eliminates dead instructions and reuses registers.
Test repeated sources, outputs retaining earlier registers, branching lifetimes,
duplicate outputs and complex constants. Exact private register numbering is
not a behavioral contract. Optimization returns a new program and must not
mutate mappings/instructions of its input.

`fuse(*programs)` concatenates outputs in argument order and offsets registers;
it requires identical parameter and runtime mappings and at least one program.
`remap_parameters(program, mapping)` adapts local theta indices to a canonical
contiguous mapping, including unused canonical entries, and shifts runtime
inputs accordingly. Missing required parameters fail. `remap_runtime_symbols`
similarly rewrites symbol slots and requires all existing keys. Both preserve
outputs, constants, source programs and arithmetic. Exercise reordered indices
and direct input outputs, not only instruction operands.

## Numerical checks and existing test migration

Retain analytic arithmetic and sinc limit tests, eager/JIT values, gradients,
Jacobians, Hessians and representative nested derivatives using backend wrappers.
Compare optimized execution with an independent analytical expression as well
as unoptimized execution. Test mixed real/complex and array runtime roots.
Concrete zero-dimensional NumPy/backend scalar operands are normalized to
constants; non-scalar arrays and active tracers must not become graph constants.

Replace old compile_ast/CompiledAST/Opcode tests with the new Program contract.
Remove tests of deleted cloning and packed-array APIs, preserving their relevant
snapshot and differentiation intent through Program tests.

## Ambiguities and suspected defects

- Compiler currently does not defensively validate directly mutated Parameter
  bounds/values; old tests expected it. Keep this as a reported contract gap,
  not a claim that invalid physics or nonfinite constants are universally banned
  (infinite optical radii are legitimate).
- Runtime array ownership is not promised: native tuple roots can alias bindings.
  Do not carry over old packed-array copy assertions.
- Runtime Symbols ignore defaults by design. Direct evaluation uses defaults.
- Optimization currently includes allocation in optimize; there is no public
  finalize. Feeding already allocated programs back into SSA-only passes is not
  documented; report this uncertainty instead of assuming idempotence.
- remap_parameters has an apparent precedence defect; add a real regression,
  leaving implementation unchanged. Arbitrary/sparse runtime-index maps lack an
  explicit validation contract. Do not invent one.
