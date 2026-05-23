from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from engram.utils.embeddings import get_embedding


class EmbeddingCache:
    """Persistent cache for embedding vectors.

    Computes an embedding once and stores it in SQLite.
    Every subsequent request for the same text returns
    the cached result instantly — no recomputation.

    At 50ms per embedding computation:
        Without cache: 1000 memories × 50ms = 50 seconds per query
        With cache:    1000 memories × 0.1ms = 0.1 seconds per query

    Usage
    -----
        cache     = EmbeddingCache()
        embedding = cache.get("I am building Engram")
        # second call returns instantly from cache
        embedding = cache.get("I am building Engram")
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash  TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                embedding  TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, text: str) -> list[float]:
        """Return embedding for text — from cache or freshly computed.

        Parameters
        ----------
        text: The text to embed.

        Returns
        -------
        768-dimension embedding vector.
        """
        text_hash = self._hash(text)

        # Check cache first
        row = self._conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
            (text_hash,)
        ).fetchone()

        if row:
            return json.loads(row[0])

        # Not cached — compute and store
        embedding = get_embedding(text)
        self._conn.execute(
            "INSERT INTO embedding_cache VALUES (?, ?, ?)",
            (text_hash, text, json.dumps(embedding))
        )
        self._conn.commit()
        return embedding

    def exists(self, text: str) -> bool:
        """Check if an embedding is already cached."""
        row = self._conn.execute(
            "SELECT 1 FROM embedding_cache WHERE text_hash = ?",
            (self._hash(text),)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        """Return number of cached embeddings."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM embedding_cache"
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    # Internal helpers

    @staticmethod
    def _hash(text: str) -> str:
        """Stable hash of text used as cache key.

        Uses SHA256 — collision resistant, consistent across runs.
        """
        import hashlib
        return hashlib.sha256(text.strip().encode()).hexdigest()