
from __future__ import annotations

import threading

from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.embeddings import cosine_similarity


_DECISION_ANCHOR = (
    "I made an important decision that will change my life forever"
)
_EMOTION_ANCHOR = (
    "I felt a strong emotion that I will never forget"
)
_CHANGE_ANCHOR = (
    "This was a turning point that changed everything for me"
)
_ACHIEVEMENT_ANCHOR = (
    "I accomplished something significant and meaningful"
)
_FRUSTRATION_ANCHOR = (
    "I was deeply frustrated and needed to do something about it"
)
_NEUTRAL_ANCHOR = (
    "Something happened today and I observed it"
)

SIGNIFICANT_ANCHORS = [
    _DECISION_ANCHOR,
    _EMOTION_ANCHOR,
    _CHANGE_ANCHOR,
    _ACHIEVEMENT_ANCHOR,
    _FRUSTRATION_ANCHOR,
]

_NORMALISE_SHIFT = 0.30
_NORMALISE_SCALE = 0.60


class SignificanceScorer:
    """Scores how significant a memory is using semantic anchor similarity.

    Unlike word-list approaches this works dynamically — it understands
    meaning not surface form. Works across languages, phrasings, and
    contexts without any hardcoded vocabulary.

    Thread-safe — anchor embeddings loaded exactly once via lock.

    Algorithm:
        1. Embed the memory and all anchors (cached)
        2. Measure similarity to each significant anchor
        3. Measure similarity to neutral anchor
        4. Significance = max(significant) - neutral_similarity
        5. Normalise to [0, 1]

    Usage:
        scorer = SignificanceScorer()
        score  = scorer.score("I decided to quit my job and build Engram")
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._cache              = EmbeddingCache(db_path)
        self._anchor_embeddings: dict[str, list[float]] | None = None
        self._neutral_embedding: list[float] | None            = None
        self._lock               = threading.Lock()

    def _load_anchors(self) -> None:
        """Load anchor embeddings exactly once — thread-safe."""
        if self._anchor_embeddings is not None:
            return
        with self._lock:
            if self._anchor_embeddings is not None:
                return
            self._anchor_embeddings = {
                anchor: self._cache.get(anchor)
                for anchor in SIGNIFICANT_ANCHORS
            }
            self._neutral_embedding = self._cache.get(_NEUTRAL_ANCHOR)

    def score(self, text: str) -> float:
        """Compute significance score for a memory.

        Parameters
        ----------
        text: The memory content to score.

        Returns
        -------
        Significance score in [0, 1].
        0.0 = completely routine
        1.0 = highly significant
        """
        if not text or not text.strip():
            return 0.0

        self._load_anchors()

        text_embedding     = self._cache.get(text)
        neutral_similarity = cosine_similarity(
            text_embedding,
            self._neutral_embedding,
        )

        max_significant = max(
            cosine_similarity(text_embedding, anchor_emb)
            for anchor_emb in self._anchor_embeddings.values()
        )

        raw = max_significant - neutral_similarity
        normalised = (raw + _NORMALISE_SHIFT) / _NORMALISE_SCALE
        return max(0.0, min(1.0, normalised))

    def score_batch(self, texts: list[str]) -> dict[str, float]:
        """Score multiple memories efficiently.

        Parameters
        ----------
        texts: List of memory contents to score.

        Returns
        -------
        Dict mapping text -> significance score.
        """
        if not texts:
            return {}
        self._load_anchors()
        return {text: self.score(text) for text in texts}