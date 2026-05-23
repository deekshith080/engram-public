from datetime import datetime, timedelta, timezone
from engram.core.memory import MemoryNode, MemoryType, MemoryStatus
from engram.core.evaluation import Evaluator, EvaluationResult
from engram.scheduler.decay import DecayReport


def make_report(active: int = 0, decaying: int = 0, pruned: int = 0) -> DecayReport:
    report = DecayReport()
    report.total_active   = active
    report.total_decaying = decaying
    report.total_pruned   = pruned
    report.total_evaluated = active + decaying + pruned
    return report


def test_evaluation_result_summary_returns_string():
    result = EvaluationResult()
    assert isinstance(result.summary(), str)
    assert "Engram Evaluation Report" in result.summary()


def test_forgetting_quality_perfect_when_nothing_pruned():
    evaluator = Evaluator()
    nodes     = [
        MemoryNode(content="I am building Engram", irreplaceability=0.9),
        MemoryNode(content="I prefer Python",      irreplaceability=0.8),
    ]
    report = make_report(active=2)
    result = evaluator.evaluate(nodes, report, [])
    assert result.forgetting_quality == 1.0


def test_forgetting_quality_higher_when_pruning_low_value():
    evaluator = Evaluator()

    kept   = MemoryNode(content="I am building Engram", irreplaceability=0.9)
    pruned = MemoryNode(content="What is Python",       irreplaceability=0.1)
    pruned.status = MemoryStatus.PRUNED

    nodes  = [kept, pruned]
    report = make_report(active=1, pruned=1)
    result = evaluator.evaluate(nodes, report, [])

    assert result.irreplaceability_kept   > result.irreplaceability_pruned
    assert result.forgetting_quality      > 0.5


def test_irreplaceability_gap_computed_correctly():
    evaluator = Evaluator()

    kept   = MemoryNode(content="Personal memory", irreplaceability=0.9)
    pruned = MemoryNode(content="Generic fact",    irreplaceability=0.1)
    pruned.status = MemoryStatus.PRUNED

    nodes  = [kept, pruned]
    report = make_report(active=1, pruned=1)
    result = evaluator.evaluate(nodes, report, [])

    assert result.irreplaceability_kept   == 0.9
    assert result.irreplaceability_pruned == 0.1


def test_notes_are_populated():
    evaluator = Evaluator()
    nodes     = [MemoryNode(content="I am building Engram", irreplaceability=0.9)]
    report    = make_report(active=1)
    result    = evaluator.evaluate(nodes, report, [])
    assert len(result.notes) > 0


def test_total_counts_match_report():
    evaluator = Evaluator()
    nodes     = [MemoryNode(content="I am building Engram")]
    report    = make_report(active=1, pruned=2, decaying=1)
    result    = evaluator.evaluate(nodes, report, [])
    assert result.total_pruned   == 2
    assert result.total_active   == 1
    assert result.total_decaying == 1