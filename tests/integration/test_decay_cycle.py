from datetime import datetime, timedelta, timezone
from engram.core.memory import MemoryNode, MemoryType, MemoryStatus
from engram.graph.manager import GraphManager, RelationshipType
from engram.scheduler.decay import DecayScheduler
from engram.scoring.engine import ScoringConfig
from engram.utils.store import InMemoryStore


def make_store_and_graph():
    return InMemoryStore(), GraphManager()


def test_high_value_memory_survives():
    store, graph = make_store_and_graph()

    keeper = MemoryNode(
        content          = "User goal is to build Engram",
        memory_type      = MemoryType.PERSONAL,
        irreplaceability = 0.95,
        access_count     = 25,
    )
    store.save(keeper)
    graph.add_node(keeper)

    extra = MemoryNode(content="Engram uses graph memory")
    store.save(extra)
    graph.add_node(extra)
    graph.add_edge(keeper.id, extra.id, RelationshipType.SEMANTIC)

    scheduler = DecayScheduler(store, graph)
    report    = scheduler.run()

    updated = store.get(keeper.id)
    assert updated.status != MemoryStatus.PRUNED


def test_low_value_memory_decays():
    store, graph = make_store_and_graph()

    weak = MemoryNode(
        content          = "What is Python",
        memory_type      = MemoryType.FACTUAL,
        irreplaceability = 0.05,
        access_count     = 0,
    )
    weak.last_accessed_at = datetime.now(timezone.utc) - timedelta(days=60)
    store.save(weak)
    graph.add_node(weak)

    scheduler = DecayScheduler(store, graph)
    report    = scheduler.run()

    updated = store.get(weak.id)
    assert updated.status in (MemoryStatus.DECAYING, MemoryStatus.PRUNED)


def test_archived_memory_never_pruned():
    store, graph = make_store_and_graph()

    pinned = MemoryNode(
        content          = "Critical rule never delete",
        irreplaceability = 0.0,
        access_count     = 0,
    )
    pinned.last_accessed_at = datetime.now(timezone.utc) - timedelta(days=365)
    pinned.pin()
    store.save(pinned)
    graph.add_node(pinned)

    scheduler = DecayScheduler(store, graph)
    scheduler.run()

    updated = store.get(pinned.id)
    assert updated.status == MemoryStatus.ARCHIVED


def test_dry_run_changes_nothing():
    store, graph = make_store_and_graph()

    node = MemoryNode(
        content          = "Temporary note",
        irreplaceability = 0.0,
        access_count     = 0,
    )
    node.last_accessed_at = datetime.now(timezone.utc) - timedelta(days=90)
    store.save(node)
    graph.add_node(node)

    scheduler = DecayScheduler(store, graph, dry_run=True)
    scheduler.run()

    updated = store.get(node.id)
    assert updated.status == MemoryStatus.ACTIVE


def test_report_counts_are_accurate():
    store, graph = make_store_and_graph()

    strong = MemoryNode(
        content          = "Strong memory",
        irreplaceability = 0.95,
        access_count     = 30,
    )
    store.save(strong)
    graph.add_node(strong)

    scheduler = DecayScheduler(store, graph)
    report    = scheduler.run()

    assert report.total_evaluated >= 1
    assert report.total_pruned + report.total_decaying + report.total_active == report.total_evaluated