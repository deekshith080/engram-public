# PRIVATE — core IP, do not share or open source
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from engram.core.consolidation import MemoryConsolidator
from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayReport, DecayScheduler, MemoryStore
from engram.scoring.engine import ScoringConfig
from engram.scoring.significance import SignificanceScorer
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.embeddings import cosine_similarity


STRENGTHEN_THRESHOLD      = 0.75
STRENGTHEN_BOOST          = 0.05
LINK_SIMILARITY_THRESHOLD = 0.72
MAX_NEW_EDGES_PER_NODE    = 3


@dataclass
class SleepReport:
    """Full report of what happened during one sleep cycle."""
    run_at:                datetime          = field(default_factory=lambda: datetime.now(timezone.utc))
    memories_strengthened: int              = 0
    memories_faded:        int              = 0
    memories_consolidated: int              = 0
    new_edges_created:     int              = 0
    decay_report:          DecayReport | None = None

    def summary(self) -> str:
        return (
            f"[{self.run_at.isoformat()}] "
            f"strengthened={self.memories_strengthened} "
            f"faded={self.memories_faded} "
            f"consolidated={self.memories_consolidated} "
            f"new_edges={self.new_edges_created}"
        )


class SleepConsolidator:
    """Runs the full sleep consolidation cycle.

    This is the brain's hippocampal consolidation process —
    running periodically to strengthen what matters,
    fade what doesn't, link related memories,
    and merge redundant fragments.

    The system gets smarter over time without user intervention.

    Cycle:
        1. Decay      — fade weak memories toward pruning
        2. Strengthen — boost high-value memories
        3. Link       — create missing semantic edges
        4. Consolidate — merge redundant memories

    Security:
        - Only touches ACTIVE memories — ARCHIVED are never modified
        - Score boost is capped at 1.0 — cannot inflate beyond max
        - Max edges per node prevents graph explosion
        - All inputs validated by upstream Pydantic models

    Usage:
        consolidator = SleepConsolidator(store, graph)
        report       = consolidator.run()
        print(report.summary())
    """

    def __init__(
        self,
        store:   MemoryStore,
        graph:   GraphManager,
        config:  ScoringConfig | None = None,
        db_path: str                  = "engram.db",
    ) -> None:
        self._store        = store
        self._graph        = graph
        self._config       = config or ScoringConfig()
        self._db_path      = db_path
        self._cache        = EmbeddingCache(db_path)
        self._sig_scorer   = SignificanceScorer(db_path)
        self._consolidator = MemoryConsolidator(db_path)

    def run(self) -> SleepReport:
        """Execute one full sleep consolidation cycle.

        Returns
        -------
        SleepReport summarising all changes made.
        """
        report = SleepReport()

        # Step 1 — Decay: fade weak memories
        scheduler           = DecayScheduler(self._store, self._graph, self._config)
        report.decay_report = scheduler.run()
        report.memories_faded = report.decay_report.total_pruned

        # Step 2 — Strengthen: boost high-value memories
        report.memories_strengthened = self._strengthen()

        # Step 3 — Link: create missing semantic edges
        report.new_edges_created = self._link_related()

        # Step 4 — Consolidate: merge redundant memories
        active = [
            n for n in self._store.get_all()
            if n.status == MemoryStatus.ACTIVE
        ]
        updated, consolidation_result = self._consolidator.consolidate(active)
        report.memories_consolidated  = consolidation_result.memories_merged

        # Save all updated nodes back to store and graph
        for node in updated:
            self._store.save(node)
            if node.status == MemoryStatus.ACTIVE:
                if not self._graph.has_node(node.id):
                    self._graph.add_node(node)

        return report

    def _strengthen(self) -> int:
        """Boost composite scores of high-value memories.

        Memories with high significance or irreplaceability
        get a small score boost — reinforcing what matters most.
        Like the brain replaying important memories during sleep.

        Score boost is always capped at 1.0.
        Only ACTIVE memories are strengthened.
        ARCHIVED memories are never touched.
        """
        strengthened = 0
        nodes        = self._store.get_all()

        for node in nodes:
            if node.status != MemoryStatus.ACTIVE:
                continue

            significance = float(node.metadata.get("significance", 0.5))
            value        = (node.irreplaceability + significance) / 2.0

            if value >= STRENGTHEN_THRESHOLD:
                node.composite_score = min(1.0, node.composite_score + STRENGTHEN_BOOST)
                self._store.save(node)
                strengthened += 1

        return strengthened

    def _link_related(self) -> int:
        """Create missing semantic edges between related memories.

        Finds pairs of active memories with high semantic similarity
        that are not yet connected in the graph, and links them.
        Like the brain forming new associations during sleep.

        Limits edges per node to prevent graph explosion.
        Only creates edges between nodes that already exist in graph.
        """
        new_edges = 0
        nodes     = [
            n for n in self._store.get_all()
            if n.status == MemoryStatus.ACTIVE
        ]

        if len(nodes) < 2:
            return 0

        embeddings = {
            node.id: self._cache.get(node.content)
            for node in nodes
        }

        for i, node_a in enumerate(nodes):
            if not self._graph.has_node(node_a.id):
                continue

            edges_added = 0

            for node_b in nodes[i + 1:]:
                if edges_added >= MAX_NEW_EDGES_PER_NODE:
                    break

                if not self._graph.has_node(node_b.id):
                    continue

                if self._graph.has_edge(node_a.id, node_b.id):
                    continue

                similarity = cosine_similarity(
                    embeddings[node_a.id],
                    embeddings[node_b.id],
                )

                if similarity >= LINK_SIMILARITY_THRESHOLD:
                    try:
                        self._graph.add_edge(
                            node_a.id,
                            node_b.id,
                            weight       = similarity,
                            relationship = "semantic",
                        )
                        new_edges   += 1
                        edges_added += 1
                    except (KeyError, ValueError):
                        continue

        return new_edges