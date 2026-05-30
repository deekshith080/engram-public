import pytest
from engram.scoring.significance import SignificanceScorer


@pytest.fixture
def scorer() -> SignificanceScorer:
    return SignificanceScorer()


def test_significance_score_in_valid_range(scorer):
    score = scorer.score("I decided to quit my job")
    assert 0.0 <= score <= 1.0


def test_empty_string_returns_zero(scorer):
    assert scorer.score("") == 0.0


def test_whitespace_returns_zero(scorer):
    assert scorer.score("   ") == 0.0


def test_life_decision_scores_higher_than_routine(scorer):
    decision = scorer.score("I decided to quit my job and start my own company")
    routine  = scorer.score("I had a meeting today")
    assert decision > routine


def test_emotional_moment_scores_higher_than_generic(scorer):
    emotional = scorer.score("I was deeply frustrated because nothing was working")
    generic   = scorer.score("What is machine learning")
    assert emotional > generic


def test_achievement_scores_higher_than_filler(scorer):
    achievement = scorer.score("I graduated from university after years of hard work")
    filler      = scorer.score("Hi how are you")
    assert achievement > filler


def test_batch_matches_individual_scores(scorer):
    texts = [
        "I decided to build Engram",
        "I had coffee this morning",
        "What is Python",
    ]
    batch      = scorer.score_batch(texts)
    individual = {text: scorer.score(text) for text in texts}
    for text in texts:
        assert abs(batch[text] - individual[text]) < 1e-6


def test_empty_batch_returns_empty_dict(scorer):
    assert scorer.score_batch([]) == {}


def test_anchors_loaded_once(scorer):
    scorer.score("first call loads anchors")
    first_anchors = scorer._anchor_embeddings
    scorer.score("second call reuses anchors")
    assert scorer._anchor_embeddings is first_anchors