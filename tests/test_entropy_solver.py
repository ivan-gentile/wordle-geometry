"""Tests for the constrained entropy-solver upper reference (3b1b method)."""
import numpy as np
import pytest

from src.wordle import feedback, pattern_to_int, EXACT
from src.patterns import generate_pattern_matrix
from src.entropy_solver import entropy_solver_game


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade",
         "shade", "spade", "scale", "stale", "stare", "scare", "snare", "share"]


def test_entropy_solver_wins_within_budget_on_small_list():
    P = generate_pattern_matrix(WORDS)
    for secret in WORDS:
        res = entropy_solver_game(P, WORDS, secret, max_guesses=6)
        assert res.won, f"entropy solver failed on {secret}"
        assert res.guesses[-1] == secret


def test_entropy_solver_never_repeats_and_respects_budget():
    P = generate_pattern_matrix(WORDS)
    res = entropy_solver_game(P, WORDS, "crane", max_guesses=6)
    assert len(set(res.guesses)) == len(res.guesses)
    assert res.n_guesses <= 6


def test_entropy_solver_uses_few_guesses_when_easy():
    # With a small, well-separated list the solver should be efficient.
    P = generate_pattern_matrix(WORDS)
    counts = [entropy_solver_game(P, WORDS, s, 6).n_guesses for s in WORDS]
    assert np.mean(counts) < 4.0  # comfortably better than the geometry agent
