import pytest
from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.core.predictive import PredictiveMemory
from engram.graph.manager import GraphManager


def make_node(
    content:          str,
    memory_type:      MemoryType = MemoryType.PERSONAL,
    irreplaceability: float      = 0.8,
) -> MemoryNode:
    return MemoryNode(
        content          = content,
        memory_type      = memory_type,
        irreplaceability = irreplaceability,
    )


@pytest.fixture
def nodes() -> list[MemoryNode]:
    return [
        make_node("I prefer Python over every other language",   irreplaceability=0.90),
        make_node("I am building Engram an AI memory system",    irreplaceability=0.95),
        make_node("I was frustrated because AI kept forgetting", irreplaceability=0.85),
        make_node("I work best in the mornings",                 irreplaceability=0.80),
        make_node("What is the capital of France",               irreplaceability=0.10),
        make_node("The weather was nice today",                  irreplaceability=0.20),
    ]


@pytest.fixture
def graph(nodes) -> GraphManager:
    g = GraphManager()
    for node in nodes:
        g.add_node(node)
    g.add_edge(nodes[1].id, nodes[2].id, relationship="causal",   weight=0.90)
    g.add_edge(nodes[1].id, nodes[0].id, relationship="semantic", weight=0.80)
    return g


def test_returns_empty_for_empty_context(nodes, graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("", nodes, graph)
    assert results == []


def test_returns_empty_for_empty_nodes(graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("working on Python project", [], graph)
    assert results == []


def test_returns_relevant_memories_for_context(nodes, graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("working on my Python project", nodes, graph)
    assert len(results) >= 1


def test_prediction_scores_in_valid_range(nodes, graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("building AI memory system", nodes, graph)
    for r in results:
        assert 0.0 <= r.prediction_score <= 1.0


def test_results_sorted_by_score_descending(nodes, graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("building AI memory system", nodes, graph)
    scores    = [r.prediction_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_pruned_memories_not_predicted(nodes, graph):
    predictor      = PredictiveMemory()
    nodes[0].status = MemoryStatus.PRUNED
    results        = predictor.predict("Python project", nodes, graph)
    returned_ids   = {r.node.id for r in results}
    assert nodes[0].id not in returned_ids


def test_list_context_works(nodes, graph):
    predictor = PredictiveMemory()
    context   = ["working on Python", "building memory system", "frustrated with AI"]
    results   = predictor.predict(context, nodes, graph)
    assert isinstance(results, list)


def test_predict_as_context_returns_string(nodes, graph):
    predictor = PredictiveMemory()
    context   = predictor.predict_as_context("Python project", nodes, graph)
    assert isinstance(context, str)
    assert len(context) > 0


def test_predict_as_context_empty_returns_message(graph):
    predictor = PredictiveMemory()
    context   = predictor.predict_as_context("test", [], graph)
    assert "No predicted" in context


def test_reason_is_string(nodes, graph):
    predictor = PredictiveMemory()
    results   = predictor.predict("building AI memory system", nodes, graph)
    for r in results:
        assert isinstance(r.reason, str)
        assert len(r.reason) > 0