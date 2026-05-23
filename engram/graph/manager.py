from __future__ import annotations
from typing import Iterable
import networkx as nx
from engram.core.memory import MemoryNode


class RelationshipType:
    """Constants for labelling edge types between memory nodes."""
    SEMANTIC   = "semantic"
    TEMPORAL   = "temporal"
    CAUSAL     = "causal"
    EXPLICIT   = "explicit"
    REINFORCES = "reinforces"


class GraphManager:
    """Manages the directed relationship graph for the Engram memory system.

    The graph is directed because relationships have meaning:
    A --causes--> B is different from B --causes--> A.

    For connectivity scoring we consider total degree (in + out) to capture
    a node's overall importance to the graph structure.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # Node management

    def add_node(self, node: MemoryNode) -> None:
        """Register a memory node. Safe to call multiple times — idempotent."""
        if not self._graph.has_node(node.id):
            self._graph.add_node(node.id, memory_id=node.id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges. Safe if node doesn't exist."""
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    # Edge management

    def add_edge(
        self,
        source_id:    str,
        target_id:    str,
        relationship: str   = RelationshipType.SEMANTIC,
        weight:       float = 1.0,
    ) -> None:
        """Create a directed edge between two memory nodes.

        Raises
        ------
        KeyError   If either node is not registered.
        ValueError If weight is out of range.
        """
        self._assert_exists(source_id)
        self._assert_exists(target_id)
        if not (0.0 < weight <= 1.0):
            raise ValueError(f"Weight must be in (0, 1], got {weight}.")
        self._graph.add_edge(
            source_id,
            target_id,
            relationship = relationship,
            weight       = weight,
        )

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a directed edge. Safe if edge doesn't exist."""
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)

    def get_neighbours(self, node_id: str) -> list[str]:
        """Return IDs of all nodes directly connected (in or out)."""
        self._assert_exists(node_id)
        successors   = set(self._graph.successors(node_id))
        predecessors = set(self._graph.predecessors(node_id))
        return list(successors | predecessors)

    # Connectivity scoring

    def raw_connectivity(self, node_id: str) -> int:
        """Total degree (in + out) of a node."""
        self._assert_exists(node_id)
        return self._graph.degree(node_id)

    def normalised_connectivity(self, node_id: str) -> float:
        """Connectivity score normalised to [0, 1].

        Capped at 1.0 — handles large graphs where a node can have
        more connections than number_of_nodes - 1 due to multi-edges.
        """
        self._assert_exists(node_id)
        total_nodes = self._graph.number_of_nodes()
        if total_nodes <= 1:
            return 0.0
        max_possible = max(total_nodes - 1, 1)
        raw = self._graph.degree(node_id) / max_possible
        return min(raw, 1.0)

    def normalised_connectivity_batch(
        self, node_ids: list[str]
    ) -> dict[str, float]:
        """Compute normalised connectivity for multiple nodes efficiently.

        Computes max_possible once — more efficient than calling
        normalised_connectivity in a loop for large graphs.
        Always returns values in [0, 1].
        """
        if not node_ids:
            return {}
        max_possible = max(self._graph.number_of_nodes() - 1, 1)
        return {
            nid: min(self._graph.degree(nid) / max_possible, 1.0)
            for nid in node_ids
            if self._graph.has_node(nid)
        }

    # Analytics

    def most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return top_n nodes by raw degree, descending."""
        return sorted(
            self._graph.degree(),
            key     = lambda x: x[1],
            reverse = True,
        )[:top_n]

    def orphan_node_ids(self) -> list[str]:
        """Return IDs of nodes with zero connections."""
        return [n for n, d in self._graph.degree() if d == 0]

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def summary(self) -> dict[str, int]:
        return {
            "nodes":   self.node_count(),
            "edges":   self.edge_count(),
            "orphans": len(self.orphan_node_ids()),
        }


    # Internal

    def _assert_exists(self, node_id: str) -> None:
        if not self._graph.has_node(node_id):
            raise KeyError(
                f"Node '{node_id}' not found. Call add_node() first."
            )