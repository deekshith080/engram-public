from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from engram.core.memory import MemoryNode, MemoryStatus
from engram.core.retrieval import RetrievalEngine
from engram.scheduler.decay import DecayReport


@dataclass
class EvaluationResult:
    """Full evaluation report for one Engram run."""
    run_at:                  datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    total_memories:          int       = 0
    total_pruned:            int       = 0
    total_active:            int       = 0
    total_decaying:          int       = 0
    precision_at_k:          float     = 0.0
    forgetting_quality:      float     = 0.0
    irreplaceability_kept:   float     = 0.0
    irreplaceability_pruned: float     = 0.0
    notes:                   list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=== Engram Evaluation Report ===",
            f"run_at                  : {self.run_at.isoformat()}",
            f"total_memories          : {self.total_memories}",
            f"active                  : {self.total_active}",
            f"decaying                : {self.total_decaying}",
            f"pruned                  : {self.total_pruned}",
            f"precision_at_k          : {self.precision_at_k:.2f}",
            f"forgetting_quality      : {self.forgetting_quality:.2f}",
            f"irreplaceability_kept   : {self.irreplaceability_kept:.2f}",
            f"irreplaceability_pruned : {self.irreplaceability_pruned:.2f}",
            "notes                   :",
        ]
        for note in self.notes:
            lines.append(f"  - {note}")
        lines.append("================================")
        return "\n".join(lines)


class Evaluator:
    """Measures whether Engram's forgetting is intelligent.

    Three core metrics:

    1. Precision@K
       Of the top-K memories returned for a query,
       how many are actually relevant?
       Range: 0.0 (all wrong) to 1.0 (all correct)

    2. Forgetting Quality
       Are we pruning the right memories?
       Measures whether pruned memories had lower irreplaceability
       than kept memories.
       Range: 0.0 (pruning randomly) to 1.0 (pruning perfectly)

    3. Irreplaceability Gap
       Average irreplaceability of kept vs pruned memories.
       A large gap means we correctly keep unique memories
       and discard generic ones.
    """

    def __init__(self) -> None:
        self._retrieval = RetrievalEngine(top_k=5)

    def evaluate(
        self,
        all_nodes:    list[MemoryNode],
        decay_report: DecayReport,
        test_queries: list[tuple[str, list[str]]],
    ) -> EvaluationResult:
        """Run a full evaluation.

        Parameters
        ----------
        all_nodes:
            Every memory node in the system.
        decay_report:
            The report from the last DecayScheduler run.
        test_queries:
            List of (query, expected_keywords) pairs.
            Example: [("what does user prefer", ["python", "prefer"])]

        Returns
        -------
        EvaluationResult with all metrics computed.
        """
        result                = EvaluationResult()
        result.total_memories = len(all_nodes)
        result.total_pruned   = decay_report.total_pruned
        result.total_active   = decay_report.total_active
        result.total_decaying = decay_report.total_decaying

        result.precision_at_k     = self._compute_precision(all_nodes, test_queries)
        result.forgetting_quality = self._compute_forgetting_quality(all_nodes)

        kept   = [n for n in all_nodes if n.status == MemoryStatus.ACTIVE]
        pruned = [n for n in all_nodes if n.status == MemoryStatus.PRUNED]

        result.irreplaceability_kept = (
            sum(n.irreplaceability for n in kept) / len(kept)
            if kept else 0.0
        )
        result.irreplaceability_pruned = (
            sum(n.irreplaceability for n in pruned) / len(pruned)
            if pruned else 0.0
        )

        if result.irreplaceability_kept > result.irreplaceability_pruned:
            result.notes.append("keeping higher value memories than pruning — correct")
        else:
            result.notes.append("pruning higher value memories than keeping — needs review")

        if result.forgetting_quality >= 0.70:
            result.notes.append("forgetting quality is strong")
        elif result.forgetting_quality >= 0.50:
            result.notes.append("forgetting quality is acceptable")
        else:
            result.notes.append("forgetting quality needs improvement")

        if result.precision_at_k >= 0.70:
            result.notes.append("retrieval precision is strong")
        else:
            result.notes.append("retrieval precision needs improvement")

        return result

    # Private helpers

    def _compute_precision(
        self,
        nodes:   list[MemoryNode],
        queries: list[tuple[str, list[str]]],
    ) -> float:
        if not queries:
            return 0.0

        total_precision = 0.0

        for query_text, expected_keywords in queries:
            results = self._retrieval.query(query_text, nodes)
            if not results:
                continue

            hits = sum(
                1 for r in results
                if any(kw.lower() in r.node.content.lower() for kw in expected_keywords)
            )
            total_precision += hits / len(results)

        return total_precision / len(queries)

    def _compute_forgetting_quality(self, nodes: list[MemoryNode]) -> float:
        kept   = [n for n in nodes if n.status == MemoryStatus.ACTIVE]
        pruned = [n for n in nodes if n.status == MemoryStatus.PRUNED]

        if not pruned:
            return 1.0
        if not kept:
            return 0.0

        avg_kept   = sum(n.irreplaceability for n in kept)   / len(kept)
        avg_pruned = sum(n.irreplaceability for n in pruned) / len(pruned)

        if avg_kept + avg_pruned == 0:
            return 0.0

        return avg_kept / (avg_kept + avg_pruned)