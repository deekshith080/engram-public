"""
Run with:
    .venv/bin/python -m uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /v1/ingest          — add memories for a user
    POST /v1/query           — retrieve relevant memories
    POST /v1/decay           — run intelligent forgetting
    GET  /v1/memories/{uid}  — get memory summary
    GET  /health             — health check
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import generate_api_key, register_key
from api.models import HealthResponse
from api.routes import router


# Rate limiter

limiter = Limiter(key_func=get_remote_address)


# App

app = FastAPI(
    title       = "Engram API",
    description = "Intelligent persistent memory for AI systems.",
    version     = "0.1.0",
    docs_url    = "/docs",
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Register all routes under /v1
app.include_router(router, prefix="/v1")


# Health check

@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health(request: Request) -> HealthResponse:
    """Health check — no auth required. 60 requests per minute max."""
    return HealthResponse(status="ok", version="0.1.0")


# Global error handler — never expose internal errors

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions.

    Never expose internal error details to the outside world.
    Log internally, return generic message externally.
    """
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error. Please try again."},
    )


# Startup

@app.on_event("startup")
async def startup() -> None:
    """Create a test API key on startup for development."""
    test_key = "engram_test_key_12345"
    register_key(test_key, owner="development")
    print()
    print("=== Engram API Server ===")
    print(f"test API key : {test_key}")
    print("docs         : http://localhost:8000/docs")
    print("health       : http://localhost:8000/health")
    print("rate limit   : 60 requests/minute per IP")
    print("========================")
    print()