# Graph module: working reference

For the full optical workflow, see [System, Simulation and Result](system-simulation-result.md).

This document records the graph module as it exists during the compiler migration. It is a reference for future documentation, not a promise that the surrounding System and Solver APIs have already been migrated.

## Structure

An expression is a directed graph of `Node` objects. Arithmetic creates `UnaryOp` and `BinaryOp` nodes; it does not immediately calculate an answer. Nodes can be shared by several expressions. A graph can have multiple roots.

| Module | Responsibility |
| --- | --- |
| `node.py` | Expression nodes and operator overloads |
| `ops.py` | Operation identifiers and their backend numerical functions |
| `relations.py` | Symbolic comparisons and control-flow errors |
| `graph_utils.py` | Graph traversal and direct interpretation |
| `compile.py` | Translation to programs, optimization and fusion |
| `program.py` | Operands, instructions and numerical execution |
| `errors.py` | Graph traversal and compilation errors |

## Nodes

- `Literal(value)` holds a numeric constant. Ordinary scalar operands in expressions are converted to literals.
- `Parameter(value, name=None, ...)` holds a mutable real scalar. Parameters are fixed by default. `variable(...)` marks one variable; `fixed()` marks it fixed; `bound(lower_bound, upper_bound)` sets its bounds. These methods return the parameter. `variable(bounds=(lower, upper))` can also set bounds; a `None` endpoint retains the current bound.
- `Symbol(key, default=None)` refers to an external value by a hashable key. `set_default(value)` changes its optional default. `has_default` is true when the default is not `None`. Direct interpretation uses the default when no binding is supplied; graph compilation does not automatically use it.
- `InputNode(node)` is a stable handle whose underlying node can be replaced. It forwards attribute access and assignment to that node. An empty handle contains `None` and cannot be compiled or interpreted.
- `UnaryOp(op, operand)` and `BinaryOp(op, left, right)` represent operations. Normal construction uses operators such as `+`, `-`, `*`, `/`, `**` and methods such as `.sin()`, `.sqrt()`, `.maximum()` and `.minimum()`.

Optical objects declare graph-backed fields with a `Node` annotation or, for dataclasses, `parameter(...)`. Each such field is exposed as an `InputNode` handle: assigning a real scalar creates a `Parameter` inside the handle, while assigning an expression such as `b + c` stores that expression inside it. Reassignment replaces the handle's underlying node, so expressions already referring to the handle remain connected. Ordinary fields are not wrapped.

Comparisons such as `parameter >= 0` create a `Comparison`, containing a `Relation` and two nodes. Converting a symbolic comparison or a node to a Python boolean raises `SymbolicControlFlowError`; Python `if` cannot directly choose a branch from a symbolic value.

`node.is_variable` checks the graph as written; Symbols report `False`. To include node-valued Symbol bindings, use `node.has_variable_parameters(bindings)`. It follows explicit compile-time bindings and raises `KeyError` for an unbound Symbol, even if that Symbol has a default. It reports the current state of reachable Parameters, not frozen Simulation metadata or whether runtime numerical values depend on theta.

## Compilation and evaluation

```python
from diffractix.graph import Parameter, Symbol, compile_graph

x = Parameter(2.0).variable()
expression = x * 3 + Symbol("offset")
program = compile_graph((expression,))
result = program.evaluate([4.0], bindings={"offset": 1.0})
# result is a tuple containing 13.0
```

`precompile(roots, bindings={})` snapshots the reachable expression structure as a `Program` with unique virtual registers. Variable parameters are recorded in `program.variables` as `id(parameter) -> local theta index`, in traversal order. Fixed parameters become constants using their current values. A bound symbol is resolved from `bindings`; a binding can be a numerical value or another `Node`, which is compiled recursively. Unbound symbols become runtime inputs, deduplicated by key in `program.runtime_symbols`.

`optimize(program)` folds conservatively supported constants, merges equivalent computations without reordering operands, removes unused instructions and allocates reusable registers. It returns a new program. There is currently no separate public finalization step: `compile_graph()` calls `optimize(precompile(...))`.

`Program.evaluate(theta, bindings={})` accepts variable values in the program's local order and numerical runtime symbol bindings, and returns a tuple in root order. Runtime bindings are numerical inputs; expressions supplied as bindings are resolved at precompilation, not recursively evaluated by `Program.evaluate()`. The program currently requires `len(theta) == len(program.variables)`.

A program contains `Input`, `Constant` and `Register` operands, and unary and binary instructions. Inputs index a combined sequence of theta values followed by runtime symbol values. Outputs may refer directly to inputs or constants as well as registers. `str(program)` displays its inputs, instructions and outputs.

`fuse(*programs)` concatenates program outputs and instructions. Currently all programs must have identical variable and runtime-symbol mappings; otherwise fusion raises `ValueError`. Independently precompiled programs do not automatically share a canonical mapping.

## Direct interpretation

`evaluate_ast(root, bindings={}, parameter_values=None)` evaluates a single root without compiling it. Symbol bindings can be numerical values or nodes and are resolved recursively. An explicit binding takes precedence over a symbol's default. With no parameter mapping, it uses current `Parameter.value` values. If `parameter_values` is supplied, it must contain every encountered parameter, keyed by `id(parameter)`; missing values raise `KeyError`.

`walk_ast(roots, bindings={})` traverses nodes in depth-first order, visiting each node identity once and following node-valued symbol bindings. Both utilities detect cycles. Symbols without a binding or default raise `KeyError` during interpretation; empty input handles raise `UnresolvedInputError`; unsupported nodes raise `UnsupportedNodeError`.

## Migration notes

The new compiler and interpreter are still being integrated with the rest of Diffractix. Some existing consumers still import legacy names such as `compile_ast`, `CompiledAST` and `graph.utils`; those imports do not match the current graph package. The intended later design is to precompile Simulation and Solver constraints, analyze redundancy on programs, and preserve runtime symbols for optical results. That integration and a shared canonical parameter mapping are not yet established by this module.
