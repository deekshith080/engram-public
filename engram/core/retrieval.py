from __future__ import annotations

from dataclasses import dataclass

from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.vector_index import VectorIndex


@dataclass
class RetrievalResult:
    """A single memory returned by a query with its relevance score."""
    node:            MemoryNode
    relevance_score: float

    def __repr__(self) -> str:
        preview = self.node.content[:60]
        return (
            f"RetrievalResult(score={self.relevance_score:.3f}, "
            f"content='{preview}')"
        )


class RetrievalEngine:
    """Finds the most relevant memories for a given query.

    Two retrieval modes:

    1. Standard — FAISS semantic search only
       Fast. Used when no graph is available.

    2. Graph-aware — FAISS + graph traversal
       Finds semantically similar memories first.
       Then follows graph edges to find connected memories.
       Causal chains surface related context automatically.
       This is what makes Engram smarter than flat memory.
    """

    def __init__(self, top_k: int = 5, db_path: str = "engram.db") -> None:
        self._top_k    = top_k
        self._cache    = EmbeddingCache(db_path)
        self._index    = VectorIndex(self._cache)
        self._node_map: dict[str, MemoryNode] = {}

    def index_nodes(self, nodes: list[MemoryNode]) -> None:
        """Build FAISS index from memory nodes."""
        for node in nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            if node.id not in self._node_map:
                self._index.add(node.id, node.content)
                self._node_map[node.id] = node

    def query(
        self,
        query_text: str,
        nodes:      list[MemoryNode],
        graph:      GraphManager | None = None,
    ) -> list[RetrievalResult]:
        """Find the most relevant memories for a query.

        Parameters
        ----------
        query_text: The question or message to search for.
        nodes:      All memories to search through.
        graph:      Optional graph for graph-aware retrieval.
                    When provided, follows edges to find connected memories.

        Returns
        -------
        List of RetrievalResult sorted by relevance, best first.
        """
        if not query_text.strip() or not nodes:
            return []

        self.index_nodes(nodes)

        if self._index.count() == 0:
            return []

        # Step 1 — FAISS semantic search
        candidates = self._index.search(query_text, top_k=self._top_k)
        scored: dict[str, float] = {}

        for node_id, similarity in candidates:
            node = self._node_map.get(node_id)
            if not node or node.status == MemoryStatus.PRUNED:
                continue
            relevance = (0.70 * similarity) + (0.30 * node.composite_score)
            scored[node_id] = relevance

        # Step 2 — graph-aware expansion
        # Follow edges from top candidates to find connected memories
        if graph is not None:
            seed_ids = list(scored.keys())
            for seed_id in seed_ids:
                if not graph.has_node(seed_id):
                    continue
                neighbours = graph.get_neighbours(seed_id)
                for neighbour_id in neighbours:
                    if neighbour_id in scored:
                        continue
                    node = self._node_map.get(neighbour_id)
                    if not node or node.status == MemoryStatus.PRUNED:
                        continue
                    # Neighbours get a boosted score based on
                    # their connection to the seed memory
                    seed_score    = scored[seed_id]
                    neighbour_score = seed_score * 0.75
                    scored[neighbour_id] = neighbour_score

        # Build results
        results = []
        for node_id, relevance in scored.items():
            node = self._node_map.get(node_id)
            if node:
                results.append(RetrievalResult(
                    node            = node,
                    relevance_score = relevance,
                ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:self._top_k]

    def query_as_context(
        self,
        query_text: str,
        nodes:      list[MemoryNode],
        graph:      GraphManager | None = None,
    ) -> str:
        """Return relevant memories formatted as context for an LLM."""
        results = self.query(query_text, nodes, graph)

        if not results:
            return "No relevant memories found."

        lines = ["Relevant memories:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r.node.memory_type.value}] "
                f"{r.node.content} "
                f"(strength={r.node.composite_score:.2f})"
            )

        return "\n".join(lines)