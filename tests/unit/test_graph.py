import pytest
from engram.core.memory import MemoryNode
from engram.graph.manager import GraphManager, RelationshipType


def test_add_node_registers_in_graph():
    gm   = GraphManager()
    node = MemoryNode(content="Memory A")
    gm.add_node(node)
    assert gm.has_node(node.id)


def test_add_node_is_idempotent():
    gm   = GraphManager()
    node = MemoryNode(content="Memory A")
    gm.add_node(node)
    gm.add_node(node)
    assert gm.summary()["nodes"] == 1


def test_remove_node_deletes_from_graph():
    gm   = GraphManager()
    node = MemoryNode(content="Memory A")
    gm.add_node(node)
    gm.remove_node(node.id)
    assert not gm.has_node(node.id)


def test_remove_nonexistent_node_does_not_crash():
    gm = GraphManager()
    gm.remove_node("nonexistent-id")


def test_add_edge_connects_nodes():
    gm = GraphManager()
    a  = MemoryNode(content="Memory A")
    b  = MemoryNode(content="Memory B")
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(a.id, b.id, RelationshipType.SEMANTIC)
    assert gm.summary()["edges"] == 1


def test_add_edge_rejects_invalid_weight():
    gm = GraphManager()
    a  = MemoryNode(content="Memory A")
    b  = MemoryNode(content="Memory B")
    gm.add_node(a)
    gm.add_node(b)
    with pytest.raises(ValueError):
        gm.add_edge(a.id, b.id, weight=0.0)


def test_isolated_node_has_zero_connectivity():
    gm   = GraphManager()
    node = MemoryNode(content="Isolated memory")
    gm.add_node(node)
    assert gm.normalised_connectivity(node.id) == 0.0


def test_fully_connected_node_scores_one():
    gm = GraphManager()
    a  = MemoryNode(content="Memory A")
    b  = MemoryNode(content="Memory B")
    c  = MemoryNode(content="Memory C")
    for node in [a, b, c]:
        gm.add_node(node)
    gm.add_edge(a.id, b.id)
    gm.add_edge(a.id, c.id)
    assert gm.normalised_connectivity(a.id) == 1.0


def test_orphan_nodes_identified():
    gm = GraphManager()
    a  = MemoryNode(content="Memory A")
    b  = MemoryNode(content="Memory B")
    c  = MemoryNode(content="Memory C")
    for node in [a, b, c]:
        gm.add_node(node)
    gm.add_edge(a.id, b.id)
    orphans = gm.orphan_node_ids()
    assert c.id in orphans
    assert a.id not in orphans


def test_unknown_node_raises_error():
    gm = GraphManager()
    with pytest.raises(KeyError):
        gm.normalised_connectivity("nonexistent-id")