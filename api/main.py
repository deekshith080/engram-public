"""
Engram API Server

Run with:
    .venv/bin/python -m uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /v1/ingest            — add memories for a user
    POST /v1/query             — retrieve relevant memories
    POST /v1/recall            — associative cascade retrieval
    POST /v1/decay             — run intelligent forgetting
    POST /v1/predict           — predictive memory
    GET  /v1/memories/{uid}    — get memory summary
    GET  /v1/timeline/{uid}    — temporal memory timeline
    GET  /health               — health check
    GET  /                     — developer dashboard
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import register_key
from api.models import HealthResponse
from api.routes import router
from engram.utils.logger import api_logger

BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title       = "Engram API",
    description = "Intelligent persistent memory for AI systems.",
    version     = "0.1.0",
    docs_url    = "/docs",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router, prefix="/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def dashboard():
    """Serve the developer dashboard."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "ok", "message": "Engram API — dashboard not found"})


@app.middleware("http")
async def log_requests(request: Request, call_next) -> JSONResponse:
    """Log every request with method, path, status and duration.

    Never logs API keys, request body, or user memory data.
    """
    start    = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    api_logger.info("request", extra={
        "method":      request.method,
        "path":        request.url.path,
        "status":      response.status_code,
        "duration_ms": duration,
    })
    return response


@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health(request: Request) -> HealthResponse:
    """Health check — verifies core dependencies are working.

    Returns 200 ok if healthy.
    Returns 503 if any dependency is broken.
    """
    try:
        from engram.utils.embeddings import get_embedding
        embedding = get_embedding("health check")
        if len(embedding) not in (384, 768):
            raise ValueError(f"Unexpected embedding dimension: {len(embedding)}")
        return HealthResponse(status="ok", version="0.1.0")
    except Exception as exc:
        api_logger.error("health check failed", extra={"error": str(exc)})
        return JSONResponse(
            status_code = 503,
            content     = {"status": "degraded", "version": "0.1.0"},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions.

    Logs internally. Never exposes details externally.
    """
    api_logger.error("unhandled exception", extra={
        "path":  request.url.path,
        "error": type(exc).__name__,
    })
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error. Please try again."},
    )


@app.on_event("startup")
async def startup() -> None:
    """Initialize schema, register test key, log startup."""
    if os.environ.get("DATABASE_URL"):
        from engram.utils.postgres_store import init_schema
        init_schema()

    test_key = "engram_test_key_12345"
    register_key(test_key, owner="development")

    api_logger.info("Engram API started", extra={
        "version":    "0.1.0",
        "static_dir": str(STATIC_DIR),
        "static_exists": STATIC_DIR.exists(),
        "endpoints":  [
            "/v1/ingest", "/v1/query", "/v1/recall",
            "/v1/decay", "/v1/predict", "/v1/timeline",
            "/v1/memories",
        ],
    })

    print()
    print("=== Engram API Server ===")
    print(f"test API key : {test_key}")
    print(f"dashboard    : http://localhost:8000")
    print(f"docs         : http://localhost:8000/docs")
    print(f"health       : http://localhost:8000/health")
    print(f"static dir   : {STATIC_DIR}")
    print("========================")
    print()