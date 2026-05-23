from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    PERSONAL  = "personal"   # user-specific, cannot be re-fetched
    FACTUAL   = "factual"    # general knowledge, exists on the internet
    EPISODIC  = "episodic"   # a specific event or conversation
    SEMANTIC  = "semantic"   # a concept or relationship between things


class MemoryStatus(str, Enum):
    ACTIVE   = "active"     # healthy, safe
    DECAYING = "decaying"   # below warning threshold
    ARCHIVED = "archived"   # pinned by user, never delete
    PRUNED   = "pruned"     # soft deleted


class MemoryNode(BaseModel):
    id:                 str           = Field(default_factory=lambda: str(uuid.uuid4()))
    content:            str           = Field(..., min_length=1)
    memory_type:        MemoryType    = MemoryType.EPISODIC
    created_at:         datetime      = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at:   datetime      = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count:       int           = Field(default=0, ge=0)
    status:             MemoryStatus  = MemoryStatus.ACTIVE
    irreplaceability:   float         = Field(default=0.5, ge=0.0, le=1.0)
    connectivity_score: float         = Field(default=0.0, ge=0.0, le=1.0)
    composite_score:    float         = Field(default=1.0, ge=0.0)
    metadata:           dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Memory content cannot be blank.")
        return v.strip()

    def touch(self) -> None:
        """Call this every time a memory is accessed."""
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1

    def pin(self) -> None:
        """Permanently protect this memory from being pruned."""
        self.status = MemoryStatus.ARCHIVED

    def age_in_seconds(self) -> float:
        """How old is this memory in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def seconds_since_access(self) -> float:
        """How long since this memory was last used."""
        return (datetime.now(timezone.utc) - self.last_accessed_at).total_seconds()