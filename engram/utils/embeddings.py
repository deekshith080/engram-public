from __future__ import annotations

import math
import os


def get_embedding(text: str) -> list[float]:
    """Compute embedding using available backend.

    Local development  → Ollama (nomic-embed-text)
    Production server  → sentence-transformers (no Ollama needed)

    Set EMBEDDING_BACKEND=sentence_transformers to force cloud mode.
    Default is ollama for local development.
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "sentence_transformers")

    if backend == "sentence_transformers":
        return _sentence_transformers_embedding(text)
    else:
        return _ollama_embedding(text)


def _ollama_embedding(text: str) -> list[float]:
    """Compute embedding using local Ollama."""
    import ollama
    result = ollama.embeddings(
        model  = "nomic-embed-text",
        prompt = text.strip(),
    )
    return result["embedding"]


def _sentence_transformers_embedding(text: str) -> list[float]:
    """Compute embedding using sentence-transformers.

    Uses all-MiniLM-L6-v2 — fast, free, no API key needed.
    384 dimensions instead of 768 — slightly less accurate
    but works on any server without Ollama.
    """
    from sentence_transformers import SentenceTransformer
    model     = _get_st_model()
    embedding = model.encode(text.strip(), normalize_embeddings=True)
    return embedding.tolist()


# Cache the model so it loads once not every call
_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Measure how similar two embeddings are.

    Range: 0.0 (completely different) to 1.0 (identical meaning)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions must match. "
            f"Got {len(vec_a)} and {len(vec_b)}."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    raw = dot_product / (magnitude_a * magnitude_b)
    return max(0.0, min(1.0, raw))


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two pieces of text."""
    return cosine_similarity(get_embedding(text_a), get_embedding(text_b))