"""Tests for the vectorized pattern matrix P[i, j] = pattern(guess_i, secret_j)."""
import numpy as np
import pytest

from src.wordle import feedback, pattern_to_int
from src.patterns import generate_pattern_matrix


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase", "gumbo"]


def test_pattern_matrix_shape_and_dtype():
    P = generate_pattern_matrix(WORDS)
    assert P.shape == (len(WORDS), len(WORDS))
    assert P.dtype == np.uint8  # 0..242 fits in uint8
    assert P.max() < 243


def test_diagonal_is_all_green():
    P = generate_pattern_matrix(WORDS)
    all_green = pattern_to_int((2, 2, 2, 2, 2))
    assert np.all(np.diag(P) == all_green)


def test_matches_scalar_feedback_on_every_pair():
    P = generate_pattern_matrix(WORDS)
    for i, g in enumerate(WORDS):
        for j, s in enumerate(WORDS):
            expected = pattern_to_int(feedback(g, s))
            assert P[i, j] == expected, f"mismatch guess={g} secret={s}"


def test_cache_roundtrip(tmp_path):
    from src.patterns import get_pattern_matrix
    cache = tmp_path / "P.npy"
    P1 = get_pattern_matrix(WORDS, cache_path=str(cache))
    assert cache.exists()
    P2 = get_pattern_matrix(WORDS, cache_path=str(cache))  # loads from cache
    assert np.array_equal(P1, P2)
    assert np.array_equal(P1, generate_pattern_matrix(WORDS))


def test_separate_guess_and_secret_lists():
    guesses = ["crane", "slate"]
    secrets = ["crate", "speed", "abbey"]
    P = generate_pattern_matrix(guesses, secrets)
    assert P.shape == (2, 3)
    for i, g in enumerate(guesses):
        for j, s in enumerate(secrets):
            assert P[i, j] == pattern_to_int(feedback(g, s))
