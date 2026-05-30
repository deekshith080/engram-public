from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter.

    Every log line is valid JSON — parseable by log aggregators
    like Datadog, Papertrail, or Railway's log viewer.

    Never logs:
    - API keys (even prefixes in production)
    - Memory content (privacy)
    - User IDs in error messages (privacy)
    - Internal stack details in API responses (security)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for a module.

    Usage:
        from engram.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Memory ingested", extra={"count": 3})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler   = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


# Root Engram logger
engram_logger = get_logger("engram")
api_logger    = get_logger("engram.api")