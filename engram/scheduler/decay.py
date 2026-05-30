from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.scoring.engine import ScoringConfig, ScoringEngine


class MemoryStore(Protocol):
    def get_all(self) -> list[MemoryNode]: ...
    def get(self, node_id: str) -> MemoryNode | None: ...
    def save(self, node: MemoryNode) -> None: ...


@dataclass
class NodeResult:
    node_id:         str
    previous_score:  float
    new_score:       float
    previous_status: MemoryStatus
    new_status:      MemoryStatus
    pruned:          bool = False


@dataclass
class DecayReport:
    run_at:          datetime         = field(default_factory=lambda: datetime.now(timezone.utc))
    total_evaluated: int              = 0
    total_active:    int              = 0
    total_decaying:  int              = 0
    total_pruned:    int              = 0
    results:         list[NodeResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.run_at.isoformat()}] "
            f"evaluated={self.total_evaluated} "
            f"active={self.total_active} "
            f"decaying={self.total_decaying} "
            f"pruned={self.total_pruned}"
        )


class DecayScheduler:
    """Runs periodic scoring and intelligent pruning over a MemoryStore.

    Supports optional significance scoring — when a SignificanceScorer
    is provided, significance is read from node metadata and passed
    to the scoring engine for brain-inspired retention decisions.

    Archived nodes are NEVER pruned regardless of score.

    Usage
    -----
        store     = InMemoryStore()
        graph     = GraphManager()
        scheduler = DecayScheduler(store, graph)
        report    = scheduler.run()
        print(report.summary())
    """

    def __init__(
        self,
        store:   MemoryStore,
        graph:   GraphManager,
        config:  ScoringConfig | None = None,
        dry_run: bool                 = False,
    ) -> None:
        self._store   = store
        self._graph   = graph
        self._engine  = ScoringEngine(config)
        self._config  = config or ScoringConfig()
        self._dry_run = dry_run

    def run(self) -> DecayReport:
        """Execute one decay cycle and return a DecayReport."""
        report = DecayReport()
        nodes  = self._store.get_all()

        if not nodes:
            return report

        node_ids         = [n.id for n in nodes]
        connectivity_map = self._graph.normalised_connectivity_batch(node_ids)

        for node in nodes:
            if node.status in (MemoryStatus.PRUNED, MemoryStatus.ARCHIVED):
                continue

            report.total_evaluated += 1
            previous_score  = node.composite_score
            previous_status = node.status
            connectivity    = connectivity_map.get(node.id, 0.0)

            # Read significance from metadata if available
            # Falls back to 0.5 — neutral — if not set
            significance = float(
                node.metadata.get("significance", 0.5)
            )
            significance = max(0.0, min(1.0, significance))

            if self._dry_run:
                new_score  = self._engine.score(node, connectivity, significance)
                new_status = previous_status
            else:
                new_score  = self._engine.apply(node, connectivity, significance)
                new_status = node.status

            pruned = new_score <= self._config.prune_threshold

            if not self._dry_run:
                if pruned:
                    node.status = MemoryStatus.PRUNED
                    new_status  = MemoryStatus.PRUNED
                self._store.save(node)
                if pruned:
                    self._graph.remove_node(node.id)

            report.results.append(NodeResult(
                node_id         = node.id,
                previous_score  = previous_score,
                new_score       = new_score,
                previous_status = previous_status,
                new_status      = new_status,
                pruned          = pruned,
            ))

            if pruned:
                report.total_pruned += 1
            elif new_status == MemoryStatus.DECAYING:
                report.total_decaying += 1
            else:
                report.total_active += 1

        return report

    def preview(self) -> DecayReport:
        """Run without persisting changes — safe inspection mode."""
        original      = self._dry_run
        self._dry_run = True
        try:
            return self.run()
        finally:
            self._dry_run = original