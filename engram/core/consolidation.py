# PRIVATE — core IP, do not share or open source
from __future__ import annotations

from dataclasses import dataclass

from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.embeddings import cosine_similarity


CONSOLIDATION_THRESHOLD  = 0.75
MIN_MEMORIES_TO_CONSIDER = 2
MAX_MEMORIES_PER_GROUP   = 5


@dataclass
class ConsolidationGroup:
    """A group of similar memories ready to be merged."""
    memories:          list[MemoryNode]
    average_similarity: float
    memory_type:       MemoryType


@dataclass
class ConsolidationResult:
    """Result of one consolidation run."""
    groups_found:   int = 0
    memories_merged: int = 0
    memories_created: int = 0


class MemoryConsolidator:
    """Merges similar memories into stronger unified knowledge.

    This is the brain's sleep consolidation process —
    during rest, the hippocampus replays and merges similar
    memory fragments into unified long-term knowledge.

    Without consolidation:
        "I prefer Python"
        "I love Python over other languages"
        "Python is my favorite language"
        → three weak fragments, noisy retrieval

    With consolidation:
        "I strongly prefer Python as my primary programming language"
        → one strong memory, clean retrieval

    Algorithm:
        1. Find groups of highly similar memories (cosine > threshold)
        2. Merge each group into one consolidated memory
        3. Consolidated memory gets max irreplaceability of the group
        4. Original memories get marked as PRUNED
        5. Return consolidation report

    Security note: only merges memories of the same type.
    Never merges PERSONAL with FACTUAL — preserves semantic integrity.
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._cache = EmbeddingCache(db_path)

    def find_groups(
        self,
        nodes: list[MemoryNode],
    ) -> list[ConsolidationGroup]:
        """Find groups of similar memories that can be consolidated.

        Parameters
        ----------
        nodes: All active memory nodes to scan.

        Returns
        -------
        List of ConsolidationGroup — each group should be merged.
        """
        active = [
            n for n in nodes
            if n.status == MemoryStatus.ACTIVE
        ]

        if len(active) < MIN_MEMORIES_TO_CONSIDER:
            return []

        visited: set[str] = set()
        groups: list[ConsolidationGroup] = []

        for i, node_a in enumerate(active):
            if node_a.id in visited:
                continue

            group     = [node_a]
            embedding_a = self._cache.get(node_a.content)

            for node_b in active[i + 1:]:
                if node_b.id in visited:
                    continue
                if node_b.memory_type != node_a.memory_type:
                    continue
                if len(group) >= MAX_MEMORIES_PER_GROUP:
                    break

                embedding_b  = self._cache.get(node_b.content)
                similarity   = cosine_similarity(embedding_a, embedding_b)

                if similarity >= CONSOLIDATION_THRESHOLD:
                    group.append(node_b)
                    visited.add(node_b.id)

            if len(group) >= MIN_MEMORIES_TO_CONSIDER:
                visited.add(node_a.id)
                avg_sim = self._average_similarity(group)
                groups.append(ConsolidationGroup(
                    memories           = group,
                    average_similarity = avg_sim,
                    memory_type        = node_a.memory_type,
                ))

        return groups

    def consolidate(
        self,
        nodes: list[MemoryNode],
    ) -> tuple[list[MemoryNode], ConsolidationResult]:
        """Find and merge similar memory groups.

        Parameters
        ----------
        nodes: All active memory nodes.

        Returns
        -------
        Tuple of (updated node list, consolidation report).
        Updated list has merged memories added and originals marked PRUNED.
        """
        groups = self.find_groups(nodes)
        result = ConsolidationResult(groups_found=len(groups))

        if not groups:
            return nodes, result

        nodes_by_id = {n.id: n for n in nodes}
        new_nodes   = []

        for group in groups:
            merged = self._merge_group(group)
            new_nodes.append(merged)
            result.memories_merged  += len(group.memories)
            result.memories_created += 1

            for original in group.memories:
                nodes_by_id[original.id].status = MemoryStatus.PRUNED

        updated = list(nodes_by_id.values()) + new_nodes
        return updated, result

    def _merge_group(self, group: ConsolidationGroup) -> MemoryNode:
        """Merge a group of similar memories into one stronger memory.

        Merged memory:
        - Content: longest memory in the group (most informative)
        - Irreplaceability: max of all group members
        - Access count: sum of all group members
        - Composite score: max of all group members
        - Metadata: records which memories were consolidated
        """
        memories = group.memories

        best_content     = max(memories, key=lambda n: len(n.content))
        max_irrepl       = max(n.irreplaceability for n in memories)
        total_access     = sum(n.access_count for n in memories)
        max_composite    = max(n.composite_score for n in memories)
        most_recent      = max(memories, key=lambda n: n.last_accessed_at)
        earliest_created = min(memories, key=lambda n: n.created_at)

        merged = MemoryNode(
            content          = best_content.content,
            memory_type      = group.memory_type,
            created_at       = earliest_created.created_at,
            last_accessed_at = most_recent.last_accessed_at,
            access_count     = total_access,
            irreplaceability = min(max_irrepl * 1.1, 1.0),
            composite_score  = max_composite,
            metadata         = {
                "consolidated_from": [n.id for n in memories],
                "consolidation_count": len(memories),
                "avg_similarity": group.average_similarity,
            },
        )
        return merged

    def _average_similarity(self, memories: list[MemoryNode]) -> float:
        """Compute average pairwise similarity within a group."""
        if len(memories) < 2:
            return 1.0

        total      = 0.0
        count      = 0
        embeddings = [self._cache.get(n.content) for n in memories]

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total += cosine_similarity(embeddings[i], embeddings[j])
                count += 1

        return total / count if count > 0 else 0.0