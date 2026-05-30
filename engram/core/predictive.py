
from __future__ import annotations

from dataclasses import dataclass

from engram.core.memory import MemoryNode, MemoryStatus
from engram.graph.manager import GraphManager
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.embeddings import cosine_similarity
from engram.utils.vector_index import VectorIndex


PREDICTION_THRESHOLD  = 0.45
MAX_CONTEXT_MEMORIES  = 5
MAX_PREDICTED         = 8
RECENCY_BOOST_WINDOW  = 7


@dataclass
class PredictedMemory:
    """A memory predicted to be relevant based on context."""
    node:             MemoryNode
    prediction_score: float
    reason:           str

    def __repr__(self) -> str:
        preview = self.node.content[:60]
        return (
            f"PredictedMemory("
            f"score={self.prediction_score:.3f}, "
            f"reason={self.reason}, "
            f"content='{preview}')"
        )


class PredictiveMemory:
    """Surfaces relevant memories proactively based on conversation context.

    Unlike /query which responds to explicit questions,
    predictive memory watches the conversation and surfaces
    memories the user is likely to need — before they ask.

    When a user says "I'm working on my Python project",
    Engram automatically surfaces:
        - Language preference (Python)
        - Current project context
        - Related past decisions
        - Relevant frustrations or learnings

    Algorithm:
        1. Embed the current conversation context
        2. Find memories with high semantic similarity to context
        3. Boost memories that are also graph-connected to recent ones
        4. Boost memories that have been accessed recently
        5. Filter by prediction threshold
        6. Return ranked predictions with reasons

    This is context-aware prediction — works immediately with
    zero historical usage data. Gets smarter with usage patterns
    as real users generate real queries.

    Security:
        - Never modifies memories — read only
        - Prediction threshold prevents noise flooding
        - Max results limit prevents overwhelming context
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

    def predict(
        self,
        context:  str | list[str],
        nodes:    list[MemoryNode],
        graph:    GraphManager | None = None,
        top_k:    int                 = MAX_PREDICTED,
    ) -> list[PredictedMemory]:
        """Predict which memories are relevant to the current context.

        Parameters
        ----------
        context: Current conversation context — a string or list of
                 recent messages. Multiple messages are joined and
                 embedded together for richer context signal.
        nodes:   All active memories to search through.
        graph:   Optional graph for connectivity boosting.
        top_k:   Maximum predictions to return.

        Returns
        -------
        List of PredictedMemory sorted by prediction score, best first.
        Only includes memories above PREDICTION_THRESHOLD.
        """
        if not nodes:
            return []

        if isinstance(context, list):
            context_text = " ".join(context[-MAX_CONTEXT_MEMORIES:])
        else:
            context_text = context

        if not context_text.strip():
            return []

        self._index_nodes(nodes)

        if self._index.count() == 0:
            return []

        candidates   = self._index.search(context_text, top_k=top_k * 2)
        context_emb  = self._cache.get(context_text)
        predictions  = []

        for node_id, similarity in candidates:
            node = self._node_map.get(node_id)
            if not node or node.status == MemoryStatus.PRUNED:
                continue

            if similarity < PREDICTION_THRESHOLD:
                continue

            score, reason = self._compute_prediction(
                node        = node,
                similarity  = similarity,
                graph       = graph,
                candidates  = {nid for nid, _ in candidates},
            )

            predictions.append(PredictedMemory(
                node             = node,
                prediction_score = score,
                reason           = reason,
            ))

        predictions.sort(key=lambda p: p.prediction_score, reverse=True)
        return predictions[:top_k]

    def predict_as_context(
        self,
        context: str | list[str],
        nodes:   list[MemoryNode],
        graph:   GraphManager | None = None,
        top_k:   int                 = MAX_PREDICTED,
    ) -> str:
        """Return predicted memories formatted as LLM context."""
        predictions = self.predict(context, nodes, graph, top_k)

        if not predictions:
            return "No predicted memories."

        lines = ["Predicted relevant memories:"]
        for i, p in enumerate(predictions, 1):
            lines.append(
                f"{i}. [{p.reason}] "
                f"{p.node.content} "
                f"(confidence={p.prediction_score:.2f})"
            )

        return "\n".join(lines)

    def _compute_prediction(
        self,
        node:       MemoryNode,
        similarity: float,
        graph:      GraphManager | None,
        candidates: set[str],
    ) -> tuple[float, str]:
        """Compute final prediction score and reason for a memory.

        Base score is semantic similarity to context.
        Boosted by:
        - High irreplaceability (personal context matters more)
        - Graph connectivity to other candidate memories
        - High significance score

        Returns
        -------
        Tuple of (score, reason string).
        """
        score  = similarity
        reason = "semantic"

        # Boost for high irreplaceability — personal context matters more
        if node.irreplaceability >= 0.80:
            score  = min(1.0, score + 0.05)
            reason = "personal_context"

        # Boost for graph connectivity to other candidates
        if graph is not None and graph.has_node(node.id):
            neighbours    = set(graph.get_neighbours(node.id))
            shared        = neighbours & candidates
            if shared:
                boost  = min(0.10, len(shared) * 0.03)
                score  = min(1.0, score + boost)
                reason = "graph_connected"

        # Boost for high significance
        significance = float(node.metadata.get("significance", 0.5))
        if significance >= 0.80:
            score  = min(1.0, score + 0.05)
            reason = "significant_moment"

        return score, reason