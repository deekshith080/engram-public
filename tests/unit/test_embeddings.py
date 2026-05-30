import math
from engram.utils.embeddings import get_embedding, cosine_similarity, semantic_similarity


def test_embedding_returns_correct_dimensions():
    embedding = get_embedding("test sentence")
    assert len(embedding) in (384, 768)


def test_embedding_returns_floats():
    embedding = get_embedding("test sentence")
    assert all(isinstance(v, float) for v in embedding)


def test_cosine_similarity_identical_texts():
    vec = get_embedding("I am building Engram")
    score = cosine_similarity(vec, vec)
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_cosine_similarity_range():
    vec_a = get_embedding("I love coding")
    vec_b = get_embedding("the weather is nice")
    score = cosine_similarity(vec_a, vec_b)
    assert 0.0 <= score <= 1.0


def test_similar_sentences_score_higher_than_unrelated():
    similar   = semantic_similarity("I love coding", "programming is my passion")
    unrelated = semantic_similarity("I love coding", "the weather is nice today")
    assert similar > unrelated


def test_cosine_similarity_rejects_mismatched_dimensions():
    import pytest
    with pytest.raises(ValueError):
        cosine_similarity([0.1, 0.2], [0.1, 0.2, 0.3])