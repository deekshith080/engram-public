from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from engram.core.memory import MemoryNode, MemoryStatus, MemoryType


class SQLiteStore:
    """Production-grade persistent storage backend using SQLite.

    Memories survive program restarts. Every memory is stored as a row.
    The store is safe to create multiple times — it never overwrites data.
    """

    def __init__(self, db_path: str = "engram.db") -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id                 TEXT PRIMARY KEY,
                content            TEXT NOT NULL,
                memory_type        TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                last_accessed_at   TEXT NOT NULL,
                access_count       INTEGER NOT NULL DEFAULT 0,
                status             TEXT NOT NULL,
                irreplaceability   REAL NOT NULL,
                connectivity_score REAL NOT NULL,
                composite_score    REAL NOT NULL,
                metadata           TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.commit()

    def save(self, node: MemoryNode) -> None:
        self._conn.execute("""
            INSERT INTO memories VALUES (
                :id, :content, :memory_type, :created_at,
                :last_accessed_at, :access_count, :status,
                :irreplaceability, :connectivity_score,
                :composite_score, :metadata
            )
            ON CONFLICT(id) DO UPDATE SET
                content            = excluded.content,
                memory_type        = excluded.memory_type,
                last_accessed_at   = excluded.last_accessed_at,
                access_count       = excluded.access_count,
                status             = excluded.status,
                irreplaceability   = excluded.irreplaceability,
                connectivity_score = excluded.connectivity_score,
                composite_score    = excluded.composite_score,
                metadata           = excluded.metadata
        """, self._serialize(node))
        self._conn.commit()

    def get(self, node_id: str) -> MemoryNode | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (node_id,)
        ).fetchone()
        return self._deserialize(row) if row else None

    def get_all(self) -> list[MemoryNode]:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [self._deserialize(row) for row in rows]

    def delete(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (node_id,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    # Internal helpers

    def _serialize(self, node: MemoryNode) -> dict:
        return {
            "id":                 node.id,
            "content":            node.content,
            "memory_type":        node.memory_type.value,
            "created_at":         node.created_at.isoformat(),
            "last_accessed_at":   node.last_accessed_at.isoformat(),
            "access_count":       node.access_count,
            "status":             node.status.value,
            "irreplaceability":   node.irreplaceability,
            "connectivity_score": node.connectivity_score,
            "composite_score":    node.composite_score,
            "metadata":           json.dumps(node.metadata),
        }

    def _deserialize(self, row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id                 = row["id"],
            content            = row["content"],
            memory_type        = MemoryType(row["memory_type"]),
            created_at         = datetime.fromisoformat(row["created_at"]),
            last_accessed_at   = datetime.fromisoformat(row["last_accessed_at"]),
            access_count       = row["access_count"],
            status             = MemoryStatus(row["status"]),
            irreplaceability   = row["irreplaceability"],
            connectivity_score = row["connectivity_score"],
            composite_score    = row["composite_score"],
            metadata           = json.loads(row["metadata"]),
        )