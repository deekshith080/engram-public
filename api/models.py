"""
api.models
~~~~~~~~~~
Pydantic models for all API request and response bodies.
Every input validated before touching the system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Ingest a message into memory for a specific user."""
    user_id:  str = Field(..., min_length=1, max_length=128)
    content:  str = Field(..., min_length=1, max_length=10_000)
    source:   str = Field(default="api", max_length=64)


class QueryRequest(BaseModel):
    """Query relevant memories for a specific user."""
    user_id:    str = Field(..., min_length=1, max_length=128)
    query:      str = Field(..., min_length=1, max_length=1_000)
    top_k:      int = Field(default=5, ge=1, le=20)


class DecayRequest(BaseModel):
    """Run a decay cycle for a specific user."""
    user_id: str = Field(..., min_length=1, max_length=128)


# Responses


class MemoryResponse(BaseModel):
    """A single memory returned from a query."""
    id:              str
    content:         str
    memory_type:     str
    relevance_score: float
    composite_score: float


class IngestResponse(BaseModel):
    """Result of an ingest operation."""
    user_id:         str
    memories_created: int
    message:         str


class QueryResponse(BaseModel):
    """Result of a query operation."""
    user_id:  str
    query:    str
    memories: list[MemoryResponse]
    count:    int


class DecayResponse(BaseModel):
    """Result of a decay cycle."""
    user_id:         str
    total_evaluated: int
    total_pruned:    int
    total_active:    int
    message:         str


class HealthResponse(BaseModel):
    """API health check response."""
    status:  str
    version: str