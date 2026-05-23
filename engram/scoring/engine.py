from __future__ import annotations

import math
from dataclasses import dataclass

from engram.core.memory import MemoryNode


@dataclass(frozen=True)
class ScoringWeights:
    irreplaceability: float = 0.40
    connectivity:     float = 0.30
    recency:          float = 0.20
    frequency:        float = 0.10

    def __post_init__(self) -> None:
        weights = [
            self.irreplaceability,
            self.connectivity,
            self.recency,
            self.frequency,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative.")
        total = sum(weights)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}.")


@dataclass
class ScoringConfig:
    weights:                ScoringWeights = None
    recency_half_life_days: float          = 7.0
    frequency_saturation:   int            = 20
    prune_threshold:        float          = 0.20
    decay_threshold:        float          = 0.40

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = ScoringWeights()
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive.")
        if self.frequency_saturation <= 0:
            raise ValueError("frequency_saturation must be positive.")
        if not (0.0 < self.prune_threshold < self.decay_threshold <= 1.0):
            raise ValueError(
                "Must satisfy: 0 < prune_threshold < decay_threshold <= 1."
            )


class ScoringEngine:

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or ScoringConfig()

    def score(self, node: MemoryNode, normalised_connectivity: float) -> float:
        if not (0.0 <= normalised_connectivity <= 1.0):
            raise ValueError("normalised_connectivity must be in [0, 1].")
        w = self._config.weights
        return (
            w.irreplaceability * node.irreplaceability
            + w.connectivity   * normalised_connectivity
            + w.recency        * self._recency_score(node)
            + w.frequency      * self._frequency_score(node)
        )

    def apply(self, node: MemoryNode, normalised_connectivity: float) -> float:
        from engram.core.memory import MemoryStatus
        composite = self.score(node, normalised_connectivity)
        node.connectivity_score = normalised_connectivity
        node.composite_score    = composite
        if node.status != MemoryStatus.ARCHIVED:
            if composite <= self._config.prune_threshold:
                node.status = MemoryStatus.DECAYING
            elif composite <= self._config.decay_threshold:
                node.status = MemoryStatus.DECAYING
            else:
                node.status = MemoryStatus.ACTIVE
        return composite

    def breakdown(self, node: MemoryNode, normalised_connectivity: float) -> dict[str, float]:
        w = self._config.weights
        parts = {
            "irreplaceability": w.irreplaceability * node.irreplaceability,
            "connectivity":     w.connectivity     * normalised_connectivity,
            "recency":          w.recency           * self._recency_score(node),
            "frequency":        w.frequency         * self._frequency_score(node),
        }
        parts["composite"] = sum(parts.values())
        return parts

    def _recency_score(self, node: MemoryNode) -> float:
        half_life_seconds = self._config.recency_half_life_days * 86_400.0
        decay_lambda      = math.log(2) / half_life_seconds
        return math.exp(-decay_lambda * node.seconds_since_access())

    def _frequency_score(self, node: MemoryNode) -> float:
        k         = self._config.frequency_saturation
        midpoint  = k / 2.0
        steepness = 10.0 / k
        return 1.0 / (1.0 + math.exp(-steepness * (node.access_count - midpoint)))