from __future__ import annotations

import math
from dataclasses import dataclass

from engram.core.memory import MemoryNode


@dataclass(frozen=True)
class ScoringWeights:
    """Relative importance of each scoring dimension.

    All weights must be >= 0 and sum to exactly 1.0.
    """
    irreplaceability: float = 0.35
    connectivity:     float = 0.25
    significance:     float = 0.20
    recency:          float = 0.12
    frequency:        float = 0.08

    def __post_init__(self) -> None:
        weights = [
            self.irreplaceability,
            self.connectivity,
            self.significance,
            self.recency,
            self.frequency,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All scoring weights must be non-negative.")
        total = sum(weights)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.6f}."
            )


@dataclass
class ScoringConfig:
    """Runtime configuration for the scoring engine."""
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
    """Computes composite retention scores for MemoryNodes.

    Five dimensions:
    1. Irreplaceability  — how unique is this memory?
    2. Connectivity      — how many memories depend on this?
    3. Significance      — how emotionally/decisionally important?
    4. Recency           — how recently was it accessed?
    5. Frequency         — how often is it accessed?

    Significance is the new brain-inspired dimension.
    It captures emotional weight and moment importance —
    things the other dimensions cannot measure.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or ScoringConfig()

    def score(
        self,
        node:                    MemoryNode,
        normalised_connectivity: float,
        significance:            float = 0.5,
    ) -> float:
        """Return composite retention score without mutating the node.

        Parameters
        ----------
        node:                    The memory node to evaluate.
        normalised_connectivity: Connectivity from GraphManager [0, 1].
        significance:            Significance from SignificanceScorer [0, 1].
                                 Defaults to 0.5 if not provided.

        Returns
        -------
        Composite score in [0, 1]. Higher = stronger retention.
        """
        if not (0.0 <= normalised_connectivity <= 1.0):
            raise ValueError("normalised_connectivity must be in [0, 1].")
        if not (0.0 <= significance <= 1.0):
            raise ValueError("significance must be in [0, 1].")

        w = self._config.weights
        return (
            w.irreplaceability * node.irreplaceability
            + w.connectivity   * normalised_connectivity
            + w.significance   * significance
            + w.recency        * self._recency_score(node)
            + w.frequency      * self._frequency_score(node)
        )

    def apply(
        self,
        node:                    MemoryNode,
        normalised_connectivity: float,
        significance:            float = 0.5,
    ) -> float:
        """Compute score and write back to the node.

        Parameters
        ----------
        node:                    The memory node to score and mutate.
        normalised_connectivity: See score().
        significance:            See score().

        Returns
        -------
        The computed composite score.
        """
        from engram.core.memory import MemoryStatus

        composite = self.score(node, normalised_connectivity, significance)
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

    def breakdown(
        self,
        node:                    MemoryNode,
        normalised_connectivity: float,
        significance:            float = 0.5,
    ) -> dict[str, float]:
        """Return per-dimension weighted contributions for debugging.

        Returns
        -------
        Dict with keys: irreplaceability, connectivity, significance,
                        recency, frequency, composite.
        """
        w = self._config.weights
        parts = {
            "irreplaceability": w.irreplaceability * node.irreplaceability,
            "connectivity":     w.connectivity     * normalised_connectivity,
            "significance":     w.significance     * significance,
            "recency":          w.recency           * self._recency_score(node),
            "frequency":        w.frequency         * self._frequency_score(node),
        }
        parts["composite"] = sum(parts.values())
        return parts

    def _recency_score(self, node: MemoryNode) -> float:
        """Exponential decay based on time since last access."""
        half_life_seconds = self._config.recency_half_life_days * 86_400.0
        decay_lambda      = math.log(2) / half_life_seconds
        return math.exp(-decay_lambda * node.seconds_since_access())

    def _frequency_score(self, node: MemoryNode) -> float:
        """Sigmoid curve that saturates at high access counts."""
        k         = self._config.frequency_saturation
        midpoint  = k / 2.0
        steepness = 10.0 / k
        return 1.0 / (1.0 + math.exp(-steepness * (node.access_count - midpoint)))