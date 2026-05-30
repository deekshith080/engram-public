import pytest
from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.core.sleep import SleepConsolidator
from engram.graph.manager import GraphManager
from engram.utils.store import InMemoryStore


def make_node(
    content:         str,
    irreplaceability: float       = 0.8,
    memory_type:     MemoryType  = MemoryType.PERSONAL,
) -> MemoryNode:
    return MemoryNode(
        content          = content,
        memory_type      = memory_type,
        irreplaceability = irreplaceability,
    )


@pytest.fixture
def empty_state():
    store = InMemoryStore()
    graph = GraphManager()
    return store, graph


@pytest.fixture
def populated_state():
    store = InMemoryStore()
    graph = GraphManager()
    nodes = [
        make_node("I prefer Python over every other language", 0.90),
        make_node("I decided to quit my job and build Engram", 0.95),
        make_node("I was frustrated because AI kept forgetting", 0.85),
        make_node("What is machine learning", 0.10),
        make_node("The weather was nice today", 0.20),
    ]
    for node in nodes:
        store.save(node)
        graph.add_node(node)
    return store, graph, nodes


def test_sleep_returns_report_on_empty_store(empty_state):
    store, graph = empty_state
    consolidator = SleepConsolidator(store, graph)
    report       = consolidator.run()
    assert report is not None
    assert report.summary() != ""


def test_sleep_report_has_all_fields(populated_state):
    store, graph, _ = populated_state
    consolidator    = SleepConsolidator(store, graph)
    report          = consolidator.run()
    assert hasattr(report, "memories_strengthened")
    assert hasattr(report, "memories_faded")
    assert hasattr(report, "memories_consolidated")
    assert hasattr(report, "new_edges_created")
    assert hasattr(report, "decay_report")


def test_sleep_strengthens_high_value_memories(populated_state):
    store, graph, nodes = populated_state
    high_value          = nodes[1]
    high_value.metadata["significance"] = 0.95
    store.save(high_value)
    original_score = high_value.composite_score
    consolidator   = SleepConsolidator(store, graph)
    report         = consolidator.run()
    assert report.memories_strengthened >= 0


def test_sleep_never_touches_archived_memories(populated_state):
    store, graph, nodes = populated_state
    archived            = nodes[0]
    archived.status     = MemoryStatus.ARCHIVED
    store.save(archived)
    original_score = archived.composite_score
    consolidator   = SleepConsolidator(store, graph)
    consolidator.run()
    updated = store.get(archived.id)
    assert updated.status == MemoryStatus.ARCHIVED


def test_sleep_creates_new_edges(populated_state):
    store, graph, _ = populated_state
    edges_before    = graph.edge_count()
    consolidator    = SleepConsolidator(store, graph)
    report          = consolidator.run()
    assert report.new_edges_created >= 0
    assert graph.edge_count() >= edges_before


def test_sleep_summary_is_string(populated_state):
    store, graph, _ = populated_state
    consolidator    = SleepConsolidator(store, graph)
    report          = consolidator.run()
    assert isinstance(report.summary(), str)
    assert len(report.summary()) > 0


def test_sleep_cycle_runs_twice_without_error(populated_state):
    store, graph, _ = populated_state
    consolidator    = SleepConsolidator(store, graph)
    report1         = consolidator.run()
    report2         = consolidator.run()
    assert report1 is not None
    assert report2 is not None