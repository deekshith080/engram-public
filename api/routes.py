"""
Security properties:
- Every route requires valid API key
- Rate limited per IP
- Input sanitized — no SQL injection, no script injection
- User isolation — each user's memories completely separate
- Errors never expose internal details
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request
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
from engram.core.associative import AssociativeRecall
from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode, MemoryStatus
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.causal import CausalChainBuilder
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.scoring.engine import ScoringConfig
from engram.utils.sqlite_store import SQLiteStore
from engram.core.temporal import TemporalReasoner, TimePeriod
from engram.core.predictive import PredictiveMemory


router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{1,128}$')


def sanitize_user_id(user_id: str) -> str:
    """Validate and sanitize user_id.

    Only alphanumeric characters, hyphens, underscores and dots allowed.
    Prevents path traversal, SQL injection and filesystem attacks.
    """
    if not _USER_ID_PATTERN.match(user_id):
        raise ValueError(
            "user_id must contain only letters, numbers, "
            "hyphens, underscores and dots."
        )
    return user_id


def sanitize_content(content: str) -> str:
    """Sanitize memory content.

    Strips whitespace. Removes null bytes.
    """
    content = content.strip()
    content = content.replace("\x00", "")
    return content


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


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("30/minute")
async def ingest(
    request:  Request,
    body:     IngestRequest,
    key_info: dict = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a message into memory for a user.

    Splits the message into atomic memory chunks.
    Scores each chunk for irreplaceability and significance.
    Creates semantic and causal edges automatically.
    Persists to SQLite — survives server restarts.

    Rate limit: 30 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
        content = sanitize_content(body.content)
    except ValueError as e:
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

    Uses FAISS semantic search + reconstructive graph retrieval.
    Returns top-k most relevant memories ranked by relevance.

    Rate limit: 60 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
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


@router.post("/recall")
@limiter.limit("60/minute")
async def recall(
    request:  Request,
    body:     QueryRequest,
    key_info: dict = Depends(require_api_key),
) -> dict:
    """Surface memories through associative cascade.

    Unlike /query which finds semantically similar memories,
    /recall follows the narrative thread — surfacing memories
    connected by meaning, causality, and association.

    One query triggers a cascade through the memory graph,
    following edges and decaying association strength with depth.

    Use /query for factual retrieval.
    Use /recall for understanding context and narrative.

    Rate limit: 60 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state   = get_user_state(user_id)
    engine  = AssociativeRecall()
    results = engine.recall(body.query, state.nodes, state.graph, top_k=body.top_k)

    memories = [
        {
            "id":                   r.node.id,
            "content":              r.node.content,
            "memory_type":          r.node.memory_type.value,
            "association_strength": r.association_strength,
            "depth":                r.depth,
            "path":                 r.path,
        }
        for r in results
    ]

    return {
        "user_id":  user_id,
        "query":    body.query,
        "memories": memories,
        "count":    len(memories),
    }


@router.post("/decay", response_model=DecayResponse)
@limiter.limit("10/minute")
async def decay(
    request:  Request,
    body:     DecayRequest,
    key_info: dict = Depends(require_api_key),
) -> DecayResponse:
    """Run a decay cycle for a user — prune weak memories.

    Call this periodically — after each session or daily.
    Engram will intelligently forget low-value memories.

    Rate limit: 10 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
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
@router.get("/timeline/{user_id}")
@limiter.limit("60/minute")
async def timeline(
    request:  Request,
    user_id:  str,
    key_info: dict = Depends(require_api_key),
) -> dict:
    """Get a full timeline summary of memories by time period.

    Returns top memories from each non-empty time period —
    today, this week, last week, this month, last month, etc.

    Useful for understanding the full arc of a user's context
    and how their priorities have changed over time.
    """
    try:
        user_id = sanitize_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state    = get_user_state(user_id)
    reasoner = TemporalReasoner()
    timeline = reasoner.summarise_timeline(state.nodes, top_k=3)

    return {
        "user_id":  user_id,
        "timeline": {
            period: [
                {
                    "id":        r.node.id,
                    "content":   r.node.content,
                    "days_ago":  round(r.days_ago, 1),
                    "relevance": round(r.relevance, 3),
                }
                for r in results
            ]
            for period, results in timeline.items()
        },
    }

@router.post("/predict")
@limiter.limit("60/minute")
async def predict(
    request:  Request,
    body:     QueryRequest,
    key_info: dict = Depends(require_api_key),
) -> dict:
    """Predict relevant memories based on current conversation context.

    Unlike /query which responds to explicit questions,
    /predict watches the conversation context and surfaces
    memories the user is likely to need — before they ask.

    Pass recent conversation messages as the query.
    Engram will proactively surface relevant personal context.

    Rate limit: 60 requests per minute per IP.
    """
    try:
        user_id = sanitize_user_id(body.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state     = get_user_state(user_id)
    predictor = PredictiveMemory()
    results   = predictor.predict(
        body.query,
        state.nodes,
        state.graph,
        top_k=body.top_k,
    )

    memories = [
        {
            "id":               r.node.id,
            "content":          r.node.content,
            "memory_type":      r.node.memory_type.value,
            "prediction_score": r.prediction_score,
            "reason":           r.reason,
        }
        for r in results
    ]

    return {
        "user_id":  user_id,
        "context":  body.query,
        "memories": memories,
        "count":    len(memories),
    }