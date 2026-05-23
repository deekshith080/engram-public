from engram.core.memory import MemoryNode, MemoryType, MemoryStatus
from engram.core.retrieval import RetrievalEngine


def test_returns_empty_for_empty_query():
    engine = RetrievalEngine()
    node   = MemoryNode(content="I am building Engram")
    result = engine.query("", [node])
    assert result == []


def test_returns_empty_for_empty_nodes():
    engine = RetrievalEngine()
    result = engine.query("what am I building", [])
    assert result == []


def test_returns_relevant_memory():
    engine = RetrievalEngine(top_k=3)
    node   = MemoryNode(
        content          = "I am building Engram",
        memory_type      = MemoryType.PERSONAL,
        irreplaceability = 0.9,
    )
    results = engine.query("what am I building", [node])
    assert len(results) > 0
    assert results[0].node.id == node.id


def test_excludes_pruned_memories():
    engine = RetrievalEngine()
    node   = MemoryNode(content="I prefer Python")
    node.status = MemoryStatus.PRUNED
    results = engine.query("what language do I prefer", [node])
    assert len(results) == 0


def test_respects_top_k():
    engine = RetrievalEngine(top_k=2)
    nodes  = [
        MemoryNode(content="I am building Engram"),
        MemoryNode(content="I prefer Python"),
        MemoryNode(content="I hate verbose code"),
        MemoryNode(content="I work on AI memory systems"),
    ]
    results = engine.query("tell me about the user", nodes)
    assert len(results) <= 2


def test_results_sorted_by_relevance():
    engine = RetrievalEngine(top_k=5)
    nodes  = [
        MemoryNode(content="I am building Engram an AI memory system"),
        MemoryNode(content="the weather is nice today"),
    ]
    results = engine.query("what is Engram", nodes)
    if len(results) >= 2:
        assert results[0].relevance_score >= results[1].relevance_score


def test_query_as_context_returns_string():
    engine  = RetrievalEngine()
    node    = MemoryNode(content="I prefer Python")
    context = engine.query_as_context("what do I prefer", [node])
    assert isinstance(context, str)
    assert len(context) > 0


def test_query_as_context_no_results():
    engine  = RetrievalEngine()
    context = engine.query_as_context("something", [])
    assert context == "No relevant memories found."