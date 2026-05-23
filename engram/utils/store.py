from __future__ import annotations

from engram.core.memory import MemoryNode


class InMemoryStore:
    """Simple dict-backed store for development and testing.
    Replace with a database-backed implementation for production."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryNode] = {}

    def get_all(self) -> list[MemoryNode]:
        return list(self._data.values())

    def get(self, node_id: str) -> MemoryNode | None:
        return self._data.get(node_id)

    def save(self, node: MemoryNode) -> None:
        self._data[node.id] = node

    def delete(self, node_id: str) -> None:
        self._data.pop(node_id, None)

    def count(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()