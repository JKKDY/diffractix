"""Traversal and variability analysis use identity, not symbolic equality."""
import pytest
from diffractix.graph import (
    ASTCycleError, InputNode, Literal, Node, Parameter, Symbol,
    UnresolvedInputError, UnsupportedNodeError, walk_ast,
)


def test_walk_is_preorder_left_to_right_and_visits_shared_nodes_once():
    first, second = Parameter(1.0), Parameter(2.0)
    shared = first + second
    root = shared * shared
    assert [id(node) for node in walk_ast((root, second))] == [
        id(root), id(shared), id(first), id(second),
    ]


def test_walk_follows_node_bindings_but_not_numeric_bindings_or_symbol_defaults():
    parameter = Parameter(2.0).variable()
    bound, numeric = Symbol("bound"), Symbol("numeric")
    defaulted = Symbol("default", default=parameter + 1)
    handle = InputNode(parameter)
    nodes = tuple(walk_ast((bound, numeric, defaulted), {"bound": handle, "numeric": 3.0}))
    assert [id(node) for node in nodes] == [id(bound), id(handle), id(parameter), id(numeric), id(defaulted)]


def test_empty_roots_and_distinct_equal_valued_parameters():
    assert tuple(walk_ast(())) == ()
    first, second = Parameter(1.0), Parameter(1.0)
    assert [id(node) for node in walk_ast((first, second))] == [id(first), id(second)]


@pytest.mark.parametrize("root,error", [
    (1, TypeError), (InputNode(None), UnresolvedInputError), (Node(), UnsupportedNodeError),
])
def test_walk_rejects_invalid_graphs(root, error):
    with pytest.raises(error):
        tuple(walk_ast((root,)))


@pytest.mark.parametrize("kind", ["handle", "binding"])
def test_walk_detects_cycles_and_repaired_graph_can_be_walked_again(kind):
    root = InputNode(None) if kind == "handle" else Symbol("loop")
    if kind == "handle":
        root.node = root + 1
        bindings = {}
    else:
        bindings = {"loop": root + 1}
    with pytest.raises(ASTCycleError, match="Cycle"):
        tuple(walk_ast((root,), bindings))
    if kind == "handle":
        root.node = Literal(1)
    else:
        bindings["loop"] = 1.0
    assert tuple(walk_ast((root,), bindings))[0] is root


def test_variable_analysis_resolves_aliases_and_tracks_parameter_state():
    parameter = Parameter(1.0)
    root = Symbol("alias") * 3
    bindings = {"alias": Symbol("parameter"), "parameter": InputNode(parameter)}
    assert not root.is_variable
    assert not root.has_variable_parameters(bindings)
    parameter.variable()
    assert root.has_variable_parameters(bindings)
    assert not root.is_variable
    parameter.fixed()
    assert not root.has_variable_parameters(bindings)
    assert not Literal(3.0).has_variable_parameters()


@pytest.mark.parametrize("variable_first", [False, True])
def test_variable_analysis_does_not_hide_unbound_symbols_after_a_variable(variable_first):
    variable = Parameter(1.0).variable()
    unknown = Symbol("unknown", default=2.0)
    root = variable + unknown if variable_first else unknown + variable
    with pytest.raises(KeyError, match="unknown"):
        root.has_variable_parameters()
    assert root.has_variable_parameters({"unknown": 2.0})


def test_variable_analysis_rejects_invalid_dependencies_even_when_other_branch_is_variable():
    root = Parameter(1.0).variable() + InputNode(None)
    with pytest.raises(UnresolvedInputError):
        root.has_variable_parameters()
