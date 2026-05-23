from __future__ import annotations

import numpy as np
import faiss

from engram.utils.embedding_cache import EmbeddingCache


class VectorIndex:
    """Fast nearest-neighbour search using FAISS.

    Automatically detects embedding dimension from first vector added.
    Works with both 768-dim Ollama and 384-dim sentence-transformers.

    At 1,500 memories:
        Linear search → 1,500 comparisons per query
        FAISS search  → ~10 comparisons per query

    At 1,000,000 memories:
        Linear search → 1,000,000 comparisons per query
        FAISS search  → ~20 comparisons per query
    """

    def __init__(self, cache: EmbeddingCache) -> None:
        self._cache    = cache
        self._index    = None  # built lazily on first add
        self._id_map:  list[str] = []

    def _ensure_index(self, dimension: int) -> None:
        """Build FAISS index on first use with correct dimension."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(dimension)

    def add(self, node_id: str, content: str) -> None:
        """Add a memory to the index."""
        embedding = self._cache.get(content)
        vector    = self._normalise(embedding)
        self._ensure_index(vector.shape[1])
        self._index.add(vector)
        self._id_map.append(node_id)

    def add_batch(self, nodes: list[tuple[str, str]]) -> None:
        """Add multiple memories at once."""
        if not nodes:
            return

        vectors = []
        for node_id, content in nodes:
            embedding = self._cache.get(content)
            vectors.append(self._normalise(embedding)[0])
            self._id_map.append(node_id)

        matrix = np.array(vectors, dtype=np.float32)
        self._ensure_index(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find the most similar memories to a query."""
        if self._index is None or self._index.ntotal == 0:
            return []

        embedding = self._cache.get(query)
        vector    = self._normalise(embedding)
        k         = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            node_id = self._id_map[idx]
            results.append((node_id, float(score)))

        return results

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    @staticmethod
    def _normalise(embedding: list[float]) -> np.ndarray:
        """Normalise to unit length for cosine similarity."""
        vec  = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec