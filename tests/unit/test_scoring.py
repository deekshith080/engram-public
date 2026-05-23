import math
import pytest
from datetime import datetime, timedelta, timezone
from engram.core.memory import MemoryNode, MemoryType
from engram.scoring.engine import ScoringEngine, ScoringWeights, ScoringConfig


def test_score_is_between_zero_and_one():
    node   = MemoryNode(content="Some memory", irreplaceability=0.5)
    engine = ScoringEngine()
    score  = engine.score(node, normalised_connectivity=0.5)
    assert 0.0 <= score <= 1.0


def test_higher_irreplaceability_gives_higher_score():
    generic  = MemoryNode(content="Generic fact",     irreplaceability=0.1)
    personal = MemoryNode(content="Personal context", irreplaceability=0.9)
    engine   = ScoringEngine()
    assert engine.score(personal, 0.5) > engine.score(generic, 0.5)


def test_higher_connectivity_gives_higher_score():
    node   = MemoryNode(content="Some memory", irreplaceability=0.5)
    engine = ScoringEngine()
    assert engine.score(node, 0.9) > engine.score(node, 0.1)


def test_older_memory_scores_lower_recency():
    recent = MemoryNode(content="Recent memory", irreplaceability=0.5)
    old    = MemoryNode(content="Old memory",    irreplaceability=0.5)
    old.last_accessed_at = datetime.now(timezone.utc) - timedelta(days=30)
    engine = ScoringEngine()
    assert engine.score(recent, 0.5) > engine.score(old, 0.5)


def test_frequently_accessed_scores_higher():
    rare     = MemoryNode(content="Rare memory",     irreplaceability=0.5, access_count=1)
    frequent = MemoryNode(content="Frequent memory", irreplaceability=0.5, access_count=50)
    engine   = ScoringEngine()
    assert engine.score(frequent, 0.5) > engine.score(rare, 0.5)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ScoringWeights(irreplaceability=0.5, connectivity=0.5, recency=0.1, frequency=0.1)


def test_breakdown_sums_to_composite():
    node      = MemoryNode(content="Some memory", irreplaceability=0.7)
    engine    = ScoringEngine()
    breakdown = engine.breakdown(node, normalised_connectivity=0.5)
    total     = sum(v for k, v in breakdown.items() if k != "composite")
    assert math.isclose(total, breakdown["composite"], abs_tol=1e-9)


def test_invalid_connectivity_raises_error():
    node   = MemoryNode(content="Some memory")
    engine = ScoringEngine()
    with pytest.raises(ValueError):
        engine.score(node, normalised_connectivity=1.5)