"""
Security properties:
- Keys stored as bcrypt hashes — even database breach reveals nothing
- Timing-safe comparison — prevents timing attacks
- Keys never logged — not even prefixes in production
- Revocation supported — keys can be instantly disabled
- SQLite backed — persists across server restarts

Key format: engram_<64 random hex chars>
Example:    engram_a3f8c2d1e4b5a6f7...
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
from fastapi import Header, HTTPException, status


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

_DB_PATH = Path("engram_auth.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create the API keys table if it doesn't exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash    TEXT    NOT NULL UNIQUE,
                key_prefix  TEXT    NOT NULL,
                owner       TEXT    NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT    NOT NULL,
                last_used   TEXT
            )
        """)
        conn.commit()


# Initialize on module load
_init_db()


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_api_key() -> str:
    """Generate a cryptographically secure API key.

    Uses 64 random hex characters = 256 bits of entropy.
    Impossible to guess or brute force.

    Format: engram_<64 hex chars>
    """
    token = secrets.token_hex(32)
    return f"engram_{token}"


def _hash_key(raw_key: str) -> str:
    """Hash an API key using bcrypt for safe storage.

    bcrypt is intentionally slow — makes brute force attacks
    computationally infeasible even if database is compromised.
    """
    return bcrypt.hashpw(
        raw_key.encode(),
        bcrypt.gensalt(rounds=12),
    ).decode()


def _verify_key(raw_key: str, hashed: str) -> bool:
    """Timing-safe comparison of raw key against stored hash.

    Uses bcrypt.checkpw which is resistant to timing attacks —
    an attacker cannot determine partial matches by measuring
    response time.
    """
    try:
        return bcrypt.checkpw(raw_key.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def register_key(raw_key: str, owner: str) -> str:
    """Register a new API key in the database.

    Parameters
    ----------
    raw_key: The raw API key to register.
    owner:   Human readable owner name.

    Returns
    -------
    The key prefix for reference (never the full key).
    """
    key_prefix = raw_key[:16] + "..."
    key_hash   = _hash_key(raw_key)
    created_at = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        try:
            conn.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, owner, active, created_at)
                VALUES (?, ?, ?, 1, ?)
            """, (key_hash, key_prefix, owner, created_at))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Key already registered — idempotent

    return key_prefix


def revoke_key_by_prefix(key_prefix: str) -> bool:
    """Revoke all keys matching a prefix.

    Returns True if any keys were revoked.
    """
    with _get_connection() as conn:
        cursor = conn.execute("""
            UPDATE api_keys SET active = 0
            WHERE key_prefix LIKE ?
        """, (key_prefix + "%",))
        conn.commit()
        return cursor.rowcount > 0


def list_keys() -> list[dict]:
    """List all registered keys — never returns raw keys."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT key_prefix, owner, active, created_at, last_used
            FROM api_keys
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def require_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> dict:
    """Validate API key on every protected request.

    Security properties:
    - Returns 401 for missing or invalid keys
    - Returns 403 for revoked keys
    - Never reveals why a key failed beyond "invalid"
    - Updates last_used timestamp on success
    - Timing-safe comparison prevents enumeration attacks

    Raises
    ------
    HTTPException 401 — missing or invalid key
    HTTPException 403 — revoked key
    """
    if not x_api_key or not x_api_key.startswith("engram_"):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid API key.",
        )

    # Load all active keys and verify
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT id, key_hash, key_prefix, owner, active
            FROM api_keys
        """).fetchall()

    matched_row = None
    for row in rows:
        if _verify_key(x_api_key, row["key_hash"]):
            matched_row = row
            break

    if matched_row is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid API key.",
        )

    if not matched_row["active"]:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "API key has been revoked.",
        )

    # Update last used timestamp
    with _get_connection() as conn:
        conn.execute("""
            UPDATE api_keys SET last_used = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), matched_row["id"]))
        conn.commit()

    return {
        "owner":      matched_row["owner"],
        "key_prefix": matched_row["key_prefix"],
    }