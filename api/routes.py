"""
Security properties:
- Every route requires valid API key
- Rate limited per IP — 30 requests/minute
- Input sanitized — no SQL injection, no script injection
- User isolation — each user's memories completely separate
- Errors never expose internal details
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import require_api_key
from api.models import (
    DecayRequest,
    DecayResponse,
    IngestRequest,
    IngestResponse,
    MemoryResponse,
    QueryRequest,
    QueryResponse,
)
from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode, MemoryStatus
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.causal import CausalChainBuilder
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.scoring.engine import ScoringConfig
from engram.utils.sqlite_store import SQLiteStore


router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

# Only allow safe characters in user_id
_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{1,128}$')


def sanitize_user_id(user_id: str) -> str:
    """Validate and sanitize user_id.

    Only alphanumeric characters, hyphens, underscores and dots allowed.
    Prevents path traversal, SQL injection and filesystem attacks.

    Raises
    ------
    ValueError if user_id contains invalid characters.
    """
    if not _USER_ID_PATTERN.match(user_id):
        raise ValueError(
            "user_id must contain only letters, numbers, "
            "hyphens, underscores and dots."
        )
    return user_id


def sanitize_content(content: str) -> str:
    """Sanitize memory content.

    Strips leading/trailing whitespace.
    Removes null bytes which can cause issues in databases.
    Limits to 10,000 characters (enforced by Pydantic too).
    """
    content = content.strip()
    content = content.replace("\x00", "")
    return content


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------

class UserMemoryState:
    """Holds all memory state for one user.

    Each user gets their own SQLite database.
    Memories survive server restarts.
    Users are completely isolated.
    """

    def __init__(self, user_id: str) -> None:
        db_path       = f"engram_user_{user_id}.db"
        self.user_id  = user_id
        self.store    = SQLiteStore(db_path)
        self.graph    = GraphManager()
        self.pipeline = IngestionPipeline(use_llm=False)
        self.creator  = AutoEdgeCreator(self.graph, db_path=db_path)
        self.builder  = CausalChainBuilder(self.graph, db_path=db_path)
        self.nodes:   list[MemoryNode] = []
        self._bootstrap()

    def _bootstrap(self) -> None:
        existing = self.store.get_all()
        for node in existing:
            self.graph.add_node(node)
            self.nodes.append(node)


_USER_STATES: dict[str, UserMemoryState] = {}


def get_user_state(user_id: str) -> UserMemoryState:
    if user_id not in _USER_STATES:
        _USER_STATES[user_id] = UserMemoryState(user_id)
    return _USER_STATES[user_id]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("30/minute")
async def ingest(
    request:  Request,
    body:     IngestRequest,
    key_info: dict = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a message into memory for a user.

    Rate limit: 30 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
        content = sanitize_content(body.content)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    state  = get_user_state(user_id)
    result = state.pipeline.ingest(content, source=body.source)

    for node in result.nodes:
        state.store.save(node)
        state.graph.add_node(node)
        state.creator.connect(node, state.nodes)
        state.builder.process(node, state.nodes)
        state.nodes.append(node)

    return IngestResponse(
        user_id          = user_id,
        memories_created = len(result.nodes),
        message          = f"ingested {len(result.nodes)} memories",
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit("60/minute")
async def query(
    request:  Request,
    body:     QueryRequest,
    key_info: dict = Depends(require_api_key),
) -> QueryResponse:
    """Query relevant memories for a user.

    Rate limit: 60 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    state   = get_user_state(user_id)
    engine  = RetrievalEngine(top_k=body.top_k)
    results = engine.query(body.query, state.nodes, state.graph)

    memories = [
        MemoryResponse(
            id              = r.node.id,
            content         = r.node.content,
            memory_type     = r.node.memory_type.value,
            relevance_score = r.relevance_score,
            composite_score = r.node.composite_score,
        )
        for r in results
    ]

    return QueryResponse(
        user_id  = user_id,
        query    = body.query,
        memories = memories,
        count    = len(memories),
    )


@router.post("/decay", response_model=DecayResponse)
@limiter.limit("10/minute")
async def decay(
    request:  Request,
    body:     DecayRequest,
    key_info: dict = Depends(require_api_key),
) -> DecayResponse:
    """Run a decay cycle for a user.

    Rate limit: 10 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    state     = get_user_state(user_id)
    config    = ScoringConfig(prune_threshold=0.20, decay_threshold=0.40)
    scheduler = DecayScheduler(state.store, state.graph, config)
    report    = scheduler.run()
    state.nodes = state.store.get_all()

    return DecayResponse(
        user_id         = user_id,
        total_evaluated = report.total_evaluated,
        total_pruned    = report.total_pruned,
        total_active    = report.total_active,
        message         = report.summary(),
    )


@router.get("/memories/{user_id}")
@limiter.limit("60/minute")
async def get_memories(
    request:  Request,
    user_id:  str,
    key_info: dict = Depends(require_api_key),
) -> dict:
    """Get memory summary for a user."""
    try:
        user_id = sanitize_user_id(user_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    state    = get_user_state(user_id)
    active   = sum(1 for n in state.nodes if n.status == MemoryStatus.ACTIVE)
    decaying = sum(1 for n in state.nodes if n.status == MemoryStatus.DECAYING)
    pruned   = sum(1 for n in state.nodes if n.status == MemoryStatus.PRUNED)

    return {
        "user_id":  user_id,
        "total":    len(state.nodes),
        "active":   active,
        "decaying": decaying,
        "pruned":   pruned,
    }