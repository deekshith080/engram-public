import pytest
from engram.core.associative import AssociativeRecall
from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.graph.manager import GraphManager


def make_node(content: str, memory_type: MemoryType = MemoryType.PERSONAL) -> MemoryNode:
    return MemoryNode(content=content, memory_type=memory_type)


def make_connected_graph() -> tuple[list[MemoryNode], GraphManager]:
    nodes = [
        make_node("I built Engram to solve AI memory problems"),
        make_node("I was deeply frustrated because AI kept forgetting everything"),
        make_node("AI memory systems store everything as flat lists with no intelligence"),
        make_node("I decided to quit my job and focus on building Engram"),
        make_node("The weather was nice today"),
    ]
    graph = GraphManager()
    for node in nodes:
        graph.add_node(node)

    graph.add_edge(nodes[0].id, nodes[1].id, relationship="causal",   weight=0.90)
    graph.add_edge(nodes[1].id, nodes[2].id, relationship="causal",   weight=0.85)
    graph.add_edge(nodes[0].id, nodes[3].id, relationship="semantic", weight=0.80)

    return nodes, graph


def test_returns_empty_for_empty_query():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("", nodes, graph)
    assert results == []


def test_returns_empty_for_empty_nodes():
    recall  = AssociativeRecall()
    graph   = GraphManager()
    results = recall.recall("what did I build", [], graph)
    assert results == []


def test_returns_results_for_valid_query():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("what did I build", nodes, graph)
    assert len(results) >= 1


def test_seed_memory_has_depth_zero():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("I built Engram", nodes, graph)
    depths       = [r.depth for r in results]
    assert 0 in depths


def test_associated_memories_have_positive_depth():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("I built Engram", nodes, graph)
    if len(results) > 1:
        assert any(r.depth > 0 for r in results)


def test_results_sorted_by_strength_descending():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("what did I build", nodes, graph)
    strengths    = [r.association_strength for r in results]
    assert strengths == sorted(strengths, reverse=True)


def test_pruned_memories_not_returned():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    nodes[2].status = MemoryStatus.PRUNED
    results      = recall.recall("AI memory problems", nodes, graph)
    returned_ids = {r.node.id for r in results}
    assert nodes[2].id not in returned_ids


def test_association_strength_in_valid_range():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    results      = recall.recall("what did I build", nodes, graph)
    for r in results:
        assert 0.0 <= r.association_strength <= 1.0


def test_recall_as_context_returns_string():
    recall       = AssociativeRecall()
    nodes, graph = make_connected_graph()
    context      = recall.recall_as_context("what did I build", nodes, graph)
    assert isinstance(context, str)
    assert len(context) > 0


def test_recall_as_context_empty_returns_message():
    recall   = AssociativeRecall()
    graph    = GraphManager()
    context  = recall.recall_as_context("test", [], graph)
    assert "No associated" in context