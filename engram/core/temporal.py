
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from engram.core.memory import MemoryNode, MemoryStatus


class TimePeriod(Enum):
    """Named time periods for temporal queries."""
    TODAY        = "today"
    THIS_WEEK    = "this_week"
    LAST_WEEK    = "last_week"
    THIS_MONTH   = "this_month"
    LAST_MONTH   = "last_month"
    LAST_3_MONTHS = "last_3_months"
    LAST_6_MONTHS = "last_6_months"
    THIS_YEAR    = "this_year"
    OLDER        = "older"


@dataclass
class TemporalResult:
    """A memory surfaced through temporal reasoning."""
    node:       MemoryNode
    period:     TimePeriod
    days_ago:   float
    relevance:  float

    def __repr__(self) -> str:
        preview = self.node.content[:60]
        return (
            f"TemporalResult("
            f"period={self.period.value}, "
            f"days_ago={self.days_ago:.1f}, "
            f"content='{preview}')"
        )


@dataclass
class TemporalComparison:
    """Comparison of memories across two time periods."""
    period_a:        TimePeriod
    period_b:        TimePeriod
    memories_a:      list[TemporalResult]
    memories_b:      list[TemporalResult]
    themes_added:    list[str]
    themes_removed:  list[str]


class TemporalReasoner:
    """Understands when memories happened and surfaces them by time.

    Memory has a temporal dimension that no other system captures.
    A goal from 6 months ago that's no longer mentioned should be
    treated differently than a goal from yesterday.

    This module answers questions like:
        "What was I working on last month?"
        "What did I decide this week?"
        "How have my priorities changed over time?"
        "What was important 3 months ago that I've forgotten?"

    Algorithm:
        1. Classify each memory into a time period
        2. Filter by requested period or range
        3. Rank by composite score within that period
        4. For comparisons — find themes present in one period but not another

    Security:
        - Never modifies memories — read only
        - All datetime comparisons use UTC
        - No external calls
    """

    def __init__(self) -> None:
        self._now = None

    def _get_now(self) -> datetime:
        """Get current UTC time. Cached per query for consistency."""
        if self._now is None:
            self._now = datetime.now(timezone.utc)
        return self._now

    def _reset_now(self) -> None:
        """Reset time cache — call before each new query."""
        self._now = None

    def classify_period(self, node: MemoryNode) -> TimePeriod:
        """Classify a memory into a named time period based on creation time."""
        now      = self._get_now()
        created  = node.created_at

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        days_ago = (now - created).total_seconds() / 86_400.0

        if days_ago < 1:
            return TimePeriod.TODAY
        if days_ago < 7:
            return TimePeriod.THIS_WEEK
        if days_ago < 14:
            return TimePeriod.LAST_WEEK
        if days_ago < 30:
            return TimePeriod.THIS_MONTH
        if days_ago < 60:
            return TimePeriod.LAST_MONTH
        if days_ago < 90:
            return TimePeriod.LAST_3_MONTHS
        if days_ago < 180:
            return TimePeriod.LAST_6_MONTHS
        if days_ago < 365:
            return TimePeriod.THIS_YEAR
        return TimePeriod.OLDER

    def query_period(
        self,
        nodes:   list[MemoryNode],
        period:  TimePeriod,
        top_k:   int = 10,
    ) -> list[TemporalResult]:
        """Get memories from a specific time period.

        Parameters
        ----------
        nodes:  All active memories to search through.
        period: The time period to filter by.
        top_k:  Maximum results to return.

        Returns
        -------
        List of TemporalResult sorted by composite score, best first.
        """
        self._reset_now()
        now     = self._get_now()
        results = []

        for node in nodes:
            if node.status == MemoryStatus.PRUNED:
                continue

            node_period = self.classify_period(node)
            if node_period != period:
                continue

            created  = node.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            days_ago = (now - created).total_seconds() / 86_400.0

            results.append(TemporalResult(
                node      = node,
                period    = node_period,
                days_ago  = days_ago,
                relevance = node.composite_score,
            ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:top_k]

    def query_range(
        self,
        nodes:      list[MemoryNode],
        start_days: float,
        end_days:   float,
        top_k:      int = 10,
    ) -> list[TemporalResult]:
        """Get memories from a custom time range.

        Parameters
        ----------
        nodes:      All active memories.
        start_days: Start of range in days ago (older boundary).
        end_days:   End of range in days ago (newer boundary).
        top_k:      Maximum results.

        Returns
        -------
        List of TemporalResult sorted by composite score.
        """
        self._reset_now()
        now     = self._get_now()
        results = []

        if start_days < end_days:
            start_days, end_days = end_days, start_days

        for node in nodes:
            if node.status == MemoryStatus.PRUNED:
                continue

            created = node.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            days_ago = (now - created).total_seconds() / 86_400.0

            if end_days <= days_ago <= start_days:
                results.append(TemporalResult(
                    node      = node,
                    period    = self.classify_period(node),
                    days_ago  = days_ago,
                    relevance = node.composite_score,
                ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:top_k]

    def compare_periods(
        self,
        nodes:    list[MemoryNode],
        period_a: TimePeriod,
        period_b: TimePeriod,
        top_k:    int = 5,
    ) -> TemporalComparison:
        """Compare memories across two time periods.

        Surfaces what was important in one period but not another.
        Useful for understanding how priorities and context have changed.

        Parameters
        ----------
        nodes:    All active memories.
        period_a: First time period (usually more recent).
        period_b: Second time period (usually older).
        top_k:    Maximum results per period.

        Returns
        -------
        TemporalComparison with memories from each period and
        themes that appear in one but not the other.
        """
        self._reset_now()
        memories_a = self.query_period(nodes, period_a, top_k)
        memories_b = self.query_period(nodes, period_b, top_k)

        contents_a = {r.node.content[:40] for r in memories_a}
        contents_b = {r.node.content[:40] for r in memories_b}

        themes_added   = [c for c in contents_a if c not in contents_b]
        themes_removed = [c for c in contents_b if c not in contents_a]

        return TemporalComparison(
            period_a       = period_a,
            period_b       = period_b,
            memories_a     = memories_a,
            memories_b     = memories_b,
            themes_added   = themes_added,
            themes_removed = themes_removed,
        )

    def summarise_timeline(
        self,
        nodes: list[MemoryNode],
        top_k: int = 3,
    ) -> dict[str, list[TemporalResult]]:
        """Get a full timeline summary of memories by period.

        Returns top memories from each non-empty time period.
        Useful for understanding the full arc of a user's context.

        Parameters
        ----------
        nodes: All active memories.
        top_k: Maximum results per period.

        Returns
        -------
        Dict mapping period name -> list of TemporalResult.
        Only includes periods that have at least one memory.
        """
        self._reset_now()
        timeline: dict[str, list[TemporalResult]] = {}

        for period in TimePeriod:
            results = self.query_period(nodes, period, top_k)
            if results:
                timeline[period.value] = results

        return timeline