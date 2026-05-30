# PRIVATE — core IP, do not share or open source
from __future__ import annotations

from dataclasses import dataclass

from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.embeddings import cosine_similarity
from engram.utils.vector_index import VectorIndex


ASSOCIATION_DECAY     = 0.85
ASSOCIATION_FLOOR     = 0.30
MAX_ASSOCIATION_DEPTH = 4
MAX_ASSOCIATION_RESULTS = 10


@dataclass
class AssociativeResult:
    """A memory surfaced through associative recall."""
    node:               MemoryNode
    association_strength: float
    depth:              int
    path:               list[str]

    def __repr__(self) -> str:
        preview = self.node.content[:60]
        return (
            f"AssociativeResult("
            f"strength={self.association_strength:.3f}, "
            f"depth={self.depth}, "
            f"content='{preview}')"
        )


class AssociativeRecall:
    """Surfaces memories through cascading graph traversal.

    Unlike semantic search which finds the most similar memories,
    associative recall follows the narrative thread — finding memories
    connected by meaning, causality, and semantic proximity.

    One query triggers a cascade:
        "why did I build Engram?"
        → "I built Engram"           (semantic, depth 0)
        → "I was frustrated"         (causal,   depth 1)
        → "AI kept forgetting"       (causal,   depth 2)
        → "I decided to quit my job" (semantic, depth 2)

    This is how humans actually remember — not lookup, but association.

    Algorithm:
        1. FAISS finds seed memories (semantic similarity)
        2. BFS traversal through graph following edge weights
        3. Association strength decays with each hop (0.85 per hop)
        4. Traversal stops when strength drops below floor (0.30)
        5. Results ranked by association strength

    Security:
        - Max depth prevents infinite graph traversal
        - Floor threshold prevents weak noise from surfacing
        - Pruned memories never returned
        - Visited set prevents cycles
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._cache    = EmbeddingCache(db_path)
        self._index    = VectorIndex(self._cache)
        self._node_map: dict[str, MemoryNode] = {}

    def _index_nodes(self, nodes: list[MemoryNode]) -> None:
        for node in nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            if node.id not in self._node_map:
                self._index.add(node.id, node.content)
                self._node_map[node.id] = node

    def recall(
        self,
        query:    str,
        nodes:    list[MemoryNode],
        graph:    GraphManager,
        top_k:    int = MAX_ASSOCIATION_RESULTS,
    ) -> list[AssociativeResult]:
        """Surface memories through associative cascade.

        Parameters
        ----------
        query:  The trigger — a question or statement.
        nodes:  All active memories to search through.
        graph:  Memory graph for edge traversal.
        top_k:  Maximum results to return.

        Returns
        -------
        List of AssociativeResult sorted by association strength.
        """
        if not query.strip() or not nodes:
            return []

        self._index_nodes(nodes)

        if self._index.count() == 0:
            return []

        # Step 1 — find seeds via semantic search
        seed_k     = min(3, len(nodes))
        candidates = self._index.search(query, top_k=seed_k)

        if not candidates:
            return []

        # Step 2 — cascade from seeds
        visited: dict[str, AssociativeResult] = {}

        for seed_id, similarity in candidates:
            node = self._node_map.get(seed_id)
            if not node or node.status == MemoryStatus.PRUNED:
                continue

            seed_strength = similarity
            if seed_id not in visited:
                visited[seed_id] = AssociativeResult(
                    node                 = node,
                    association_strength = seed_strength,
                    depth                = 0,
                    path                 = [seed_id],
                )

            self._cascade(
                node_id  = seed_id,
                strength = seed_strength,
                depth    = 0,
                path     = [seed_id],
                graph    = graph,
                visited  = visited,
            )

        if not visited:
            return []

        results = sorted(
            visited.values(),
            key     = lambda r: r.association_strength,
            reverse = True,
        )
        return results[:top_k]

    def recall_as_context(
        self,
        query:  str,
        nodes:  list[MemoryNode],
        graph:  GraphManager,
        top_k:  int = MAX_ASSOCIATION_RESULTS,
    ) -> str:
        """Return associative recall formatted as LLM context."""
        results = self.recall(query, nodes, graph, top_k)

        if not results:
            return "No associated memories found."

        lines = ["Associated memories:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [depth={r.depth}, strength={r.association_strength:.2f}] "
                f"{r.node.content}"
            )

        return "\n".join(lines)

    def _cascade(
        self,
        node_id:  str,
        strength: float,
        depth:    int,
        path:     list[str],
        graph:    GraphManager,
        visited:  dict[str, AssociativeResult],
    ) -> None:
        """Recursively follow graph edges to surface associated memories.

        Stops when:
        - Max depth reached
        - Association strength drops below floor
        - All neighbours already visited
        """
        if depth >= MAX_ASSOCIATION_DEPTH:
            return

        if not graph.has_node(node_id):
            return

        neighbours = graph.get_neighbours(node_id)

        for neighbour_id in neighbours:
            if neighbour_id in path:
                continue

            node = self._node_map.get(neighbour_id)
            if not node or node.status == MemoryStatus.PRUNED:
                continue

            edge_weight        = self._get_edge_weight(graph, node_id, neighbour_id)
            neighbour_strength = strength * ASSOCIATION_DECAY * edge_weight

            if neighbour_strength < ASSOCIATION_FLOOR:
                continue

            new_path = path + [neighbour_id]

            if neighbour_id not in visited or \
               visited[neighbour_id].association_strength < neighbour_strength:
                visited[neighbour_id] = AssociativeResult(
                    node                 = node,
                    association_strength = neighbour_strength,
                    depth                = depth + 1,
                    path                 = new_path,
                )

            self._cascade(
                node_id  = neighbour_id,
                strength = neighbour_strength,
                depth    = depth + 1,
                path     = new_path,
                graph    = graph,
                visited  = visited,
            )

    @staticmethod
    def _get_edge_weight(
        graph:     GraphManager,
        source_id: str,
        target_id: str,
    ) -> float:
        """Get edge weight between two nodes. Defaults to 1.0."""
        try:
            edge_data = graph._graph.get_edge_data(source_id, target_id)
            if edge_data:
                return float(edge_data.get("weight", 1.0))
            edge_data = graph._graph.get_edge_data(target_id, source_id)
            if edge_data:
                return float(edge_data.get("weight", 1.0))
        except Exception:
            pass
        return 1.0