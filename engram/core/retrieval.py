from __future__ import annotations

import string
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.vector_index import VectorIndex


@dataclass
class RetrievalResult:
    """A single memory returned by a query with its relevance score."""
    node:            MemoryNode
    relevance_score: float
    source:          str = "semantic"

    def __repr__(self) -> str:
        preview = self.node.content[:60]
        return (
            f"RetrievalResult(score={self.relevance_score:.3f}, "
            f"source={self.source}, "
            f"content='{preview}')"
        )


class RetrievalEngine:
    """Finds the most relevant memories using hybrid retrieval.

    Three retrieval signals combined:

    1. FAISS semantic search
       Fast vector similarity — finds conceptually related memories.

    2. BM25 lexical search
       Keyword matching — captures exact names, dates, specific terms
       that semantic search misses. Index cached for performance.

    3. Reconstructive graph expansion
       Follows causal and semantic edges to reconstruct narrative context.

    Testing effect:
       Every retrieved memory gets a small composite score boost.
       Retrieval strengthens memory — just like the human brain.
       Boost diminishes with each access (saturation curve).

    Performance:
       BM25 index cached — only rebuilt when node list changes.
       FAISS index incremental — nodes added once, never rebuilt.
       Both indexes share the same node_map for memory efficiency.
    """

    def __init__(self, top_k: int = 5, db_path: str = "engram.db") -> None:
        self._top_k       = top_k
        self._cache       = EmbeddingCache(db_path)
        self._index       = VectorIndex(self._cache)
        self._node_map:   dict[str, MemoryNode] = {}
        self._bm25:       BM25Okapi | None      = None
        self._bm25_nodes: list[MemoryNode]      = []

    def index_nodes(self, nodes: list[MemoryNode]) -> None:
        """Build FAISS index from memory nodes — incremental."""
        for node in nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            if node.id not in self._node_map:
                self._index.add(node.id, node.content)
                self._node_map[node.id] = node

    def query(
        self,
        query_text:     str,
        nodes:          list[MemoryNode],
        graph:          GraphManager | None = None,
        reconstructive: bool               = True,
        bm25_weight:    float              = 0.30,
    ) -> list[RetrievalResult]:
        """Find the most relevant memories for a query.

        Hybrid retrieval — semantic + BM25 + graph reconstruction.

        Parameters
        ----------
        query_text:     The question or message to search for.
        nodes:          All memories to search through.
        graph:          Graph for reconstructive retrieval.
        reconstructive: Follow graph edges to reconstruct context.
        bm25_weight:    Weight for BM25 score in [0, 1].
                        0 = pure semantic, 1 = pure BM25.
                        Default 0.30 — semantic-dominant hybrid.

        Returns
        -------
        List of RetrievalResult sorted by relevance, best first.
        """
        if not query_text.strip() or not nodes:
            return []

        active_nodes = [n for n in nodes if n.status != MemoryStatus.PRUNED]
        if not active_nodes:
            return []

        self.index_nodes(active_nodes)

        if self._index.count() == 0:
            return []

        # Step 1 — FAISS semantic search
        candidates      = self._index.search(query_text, top_k=self._top_k)
        scored: dict[str, tuple[float, str]] = {}

        for node_id, similarity in candidates:
            node = self._node_map.get(node_id)
            if not node or node.status == MemoryStatus.PRUNED:
                continue
            relevance       = (0.70 * similarity) + (0.30 * node.composite_score)
            scored[node_id] = (relevance, "semantic")

        # Step 2 — BM25 lexical search
        bm25_scores = self._bm25_scores(query_text, active_nodes)
        for node_id, bm25_score in bm25_scores.items():
            if node_id in scored:
                semantic_score, _ = scored[node_id]
                hybrid            = (
                    (1.0 - bm25_weight) * semantic_score
                    + bm25_weight       * bm25_score
                )
                scored[node_id] = (hybrid, "hybrid")
            else:
                if bm25_score > 0.10:
                    node = self._node_map.get(node_id)
                    if node:
                        scored[node_id] = (bm25_weight * bm25_score, "bm25")

        # Step 3 — Reconstructive graph expansion
        if reconstructive and graph is not None:
            seed_ids = list(scored.keys())
            self._expand_via_graph(seed_ids, scored, graph)

        # Build results
        results = []
        for node_id, (relevance, source) in scored.items():
            node = self._node_map.get(node_id)
            if node:
                results.append(RetrievalResult(
                    node            = node,
                    relevance_score = relevance,
                    source          = source,
                ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        top_results = results[:self._top_k]

        # Testing effect — retrieved memories get stronger
        for r in top_results:
            r.node.touch()

        return top_results

    def query_as_context(
        self,
        query_text:     str,
        nodes:          list[MemoryNode],
        graph:          GraphManager | None = None,
        reconstructive: bool               = True,
    ) -> str:
        """Return relevant memories formatted as context for an LLM."""
        results = self.query(query_text, nodes, graph, reconstructive)

        if not results:
            return "No relevant memories found."

        lines = ["Relevant memories:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r.node.memory_type.value}] "
                f"{r.node.content} "
                f"(strength={r.node.composite_score:.2f}, via={r.source})"
            )

        return "\n".join(lines)

    def _bm25_scores(
        self,
        query_text: str,
        nodes:      list[MemoryNode],
    ) -> dict[str, float]:
        """Compute BM25 lexical scores for all nodes.

        BM25 index is cached and only rebuilt when nodes change.
        Repeated queries on the same node set are O(1) for index lookup.

        Returns
        -------
        Dict mapping node_id -> normalised BM25 score in [0, 1].
        """
        if not nodes:
            return {}

        def tokenize(text: str) -> list[str]:
            text = text.lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            return text.split()

        # Rebuild index only if node list changed
        node_ids   = [n.id for n in nodes]
        cached_ids = [n.id for n in self._bm25_nodes]
        if node_ids != cached_ids or self._bm25 is None:
            corpus           = [tokenize(n.content) for n in nodes]
            self._bm25       = BM25Okapi(corpus)
            self._bm25_nodes = list(nodes)

        query_toks = tokenize(query_text)
        scores     = self._bm25.get_scores(query_toks)
        max_score  = max(scores) if max(scores) > 0 else 1.0

        return {
            nodes[i].id: float(scores[i]) / max_score
            for i in range(len(nodes))
        }

    def _expand_via_graph(
        self,
        seed_ids: list[str],
        scored:   dict[str, tuple[float, str]],
        graph:    GraphManager,
    ) -> None:
        """Expand seed memories by following graph edges.

        Causal edges followed backwards — find WHY.
        Semantic edges followed outwards — find WHAT ELSE.
        Neighbours get a discounted score based on edge weight.
        """
        for seed_id in seed_ids:
            if not graph.has_node(seed_id):
                continue

            seed_score = scored[seed_id][0]
            neighbours = graph.get_neighbours(seed_id)

            for neighbour_id in neighbours:
                if neighbour_id in scored:
                    continue

                node = self._node_map.get(neighbour_id)
                if not node or node.status == MemoryStatus.PRUNED:
                    continue

                edge_type       = self._get_edge_type(graph, seed_id, neighbour_id)
                boost           = 0.85 if edge_type == "causal" else 0.70
                neighbour_score = seed_score * boost

                scored[neighbour_id] = (neighbour_score, edge_type)

    @staticmethod
    def _get_edge_type(
        graph:     GraphManager,
        source_id: str,
        target_id: str,
    ) -> str:
        """Get the relationship type of an edge between two nodes."""
        try:
            edge_data = graph._graph.get_edge_data(source_id, target_id)
            if edge_data:
                return edge_data.get("relationship", "semantic")
            edge_data = graph._graph.get_edge_data(target_id, source_id)
            if edge_data:
                return edge_data.get("relationship", "semantic")
        except Exception:
            pass
        return "semantic"