from datetime import datetime, timedelta, timezone

import pytest

from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.core.temporal import TemporalReasoner, TimePeriod


def make_node(content: str, days_ago: float = 0.0) -> MemoryNode:
    node = MemoryNode(content=content, memory_type=MemoryType.PERSONAL)
    node.created_at       = datetime.now(timezone.utc) - timedelta(days=days_ago)
    node.last_accessed_at = node.created_at
    return node


@pytest.fixture
def reasoner() -> TemporalReasoner:
    return TemporalReasoner()


@pytest.fixture
def timeline_nodes() -> list[MemoryNode]:
    return [
        make_node("I prefer Python",               days_ago=0.5),
        make_node("I started building Engram",     days_ago=5),
        make_node("I was frustrated with AI",      days_ago=10),
        make_node("I decided to quit my job",      days_ago=25),
        make_node("I graduated from university",   days_ago=50),
        make_node("I moved to San Francisco",      days_ago=100),
        make_node("I learned to code",             days_ago=200),
        make_node("I got my first job",            days_ago=400),
    ]


def test_classify_today(reasoner):
    node   = make_node("recent memory", days_ago=0.5)
    period = reasoner.classify_period(node)
    assert period == TimePeriod.TODAY


def test_classify_this_week(reasoner):
    node   = make_node("this week memory", days_ago=5)
    period = reasoner.classify_period(node)
    assert period == TimePeriod.THIS_WEEK


def test_classify_this_month(reasoner):
    node   = make_node("this month memory", days_ago=20)
    period = reasoner.classify_period(node)
    assert period == TimePeriod.THIS_MONTH


def test_classify_last_month(reasoner):
    node   = make_node("last month memory", days_ago=45)
    period = reasoner.classify_period(node)
    assert period == TimePeriod.LAST_MONTH


def test_classify_older(reasoner):
    node   = make_node("old memory", days_ago=400)
    period = reasoner.classify_period(node)
    assert period == TimePeriod.OLDER


def test_query_period_returns_correct_memories(reasoner, timeline_nodes):
    results = reasoner.query_period(timeline_nodes, TimePeriod.THIS_WEEK)
    assert len(results) >= 1
    for r in results:
        assert r.period == TimePeriod.THIS_WEEK


def test_query_period_excludes_pruned(reasoner, timeline_nodes):
    timeline_nodes[0].status = MemoryStatus.PRUNED
    results = reasoner.query_period(timeline_nodes, TimePeriod.TODAY)
    ids     = {r.node.id for r in results}
    assert timeline_nodes[0].id not in ids


def test_query_period_sorted_by_relevance(reasoner, timeline_nodes):
    results    = reasoner.query_period(timeline_nodes, TimePeriod.THIS_WEEK, top_k=10)
    relevances = [r.relevance for r in results]
    assert relevances == sorted(relevances, reverse=True)


def test_query_range_returns_correct_memories(reasoner, timeline_nodes):
    results = reasoner.query_range(timeline_nodes, start_days=30, end_days=0)
    for r in results:
        assert 0 <= r.days_ago <= 30


def test_query_range_empty_when_no_memories_in_range(reasoner, timeline_nodes):
    results = reasoner.query_range(timeline_nodes, start_days=500, end_days=450)
    assert len(results) == 0


def test_compare_periods_returns_comparison(reasoner, timeline_nodes):
    comparison = reasoner.compare_periods(
        timeline_nodes,
        TimePeriod.THIS_WEEK,
        TimePeriod.LAST_MONTH,
    )
    assert comparison.period_a == TimePeriod.THIS_WEEK
    assert comparison.period_b == TimePeriod.LAST_MONTH
    assert isinstance(comparison.memories_a, list)
    assert isinstance(comparison.memories_b, list)


def test_summarise_timeline_returns_dict(reasoner, timeline_nodes):
    timeline = reasoner.summarise_timeline(timeline_nodes)
    assert isinstance(timeline, dict)
    assert len(timeline) > 0


def test_summarise_timeline_only_nonempty_periods(reasoner, timeline_nodes):
    timeline = reasoner.summarise_timeline(timeline_nodes)
    for period_name, results in timeline.items():
        assert len(results) > 0


def test_days_ago_is_positive(reasoner, timeline_nodes):
    results = reasoner.query_period(timeline_nodes, TimePeriod.THIS_WEEK)
    for r in results:
        assert r.days_ago >= 0