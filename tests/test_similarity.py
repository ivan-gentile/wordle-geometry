"""Tests for Relation-B similarity S[a,b] = mean_g 1[P[g,a] == P[g,b]]."""
import numpy as np
import pytest

from src.patterns import generate_pattern_matrix
from src.similarity import relation_b_similarity


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase", "slate"]


def test_similarity_shape_and_range():
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    assert S.shape == (len(WORDS), len(WORDS))
    assert np.all(S >= 0.0) and np.all(S <= 1.0)


def test_diagonal_is_one():
    # A word is always perfectly consistent with itself: every guess gives the
    # same pattern for secret a as for secret a.
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    assert np.allclose(np.diag(S), 1.0)


def test_symmetric():
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    assert np.allclose(S, S.T)


def test_matches_direct_definition():
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    n = len(WORDS)
    for a in range(n):
        for b in range(n):
            expected = np.mean(P[:, a] == P[:, b])
            assert abs(S[a, b] - expected) < 1e-9


def test_near_words_more_similar_than_far_words():
    # CRANE and CRATE differ in one letter -> often produce the same pattern
    # under a random probe, so they should be more B-similar than CRANE/PUPPY.
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    i_crane, i_crate = WORDS.index("crane"), WORDS.index("crate")
    i_puppy = WORDS.index("puppy")
    assert S[i_crane, i_crate] > S[i_crane, i_puppy]
