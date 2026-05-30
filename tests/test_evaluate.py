"""Tests for the evaluation harness aggregation and splitting."""
import numpy as np
import pytest

from src.agents import GameResult
from src.evaluate import summarize, split_secrets


def test_summarize_win_rate_and_histogram():
    results = [
        GameResult("a", True, 3, ["x", "y", "a"], []),
        GameResult("b", True, 2, ["x", "b"], []),
        GameResult("c", False, 6, ["x"] * 6, []),
    ]
    m = summarize("t", results, max_guesses=6)
    assert m.n == 3
    assert abs(m.win_rate - 2 / 3) < 1e-9
    assert abs(m.mean_guesses_won - 2.5) < 1e-9      # (3 + 2) / 2
    # loss counted as budget + 1 = 7: mean of [3, 2, 7]
    assert abs(m.mean_guesses_all - 4.0) < 1e-9
    assert m.hist == {3: 1, 2: 1}


def test_split_is_disjoint_and_covers_all():
    words = [f"w{i:04d}" for i in range(100)]
    tune, evl = split_secrets(words, frac_tune=0.5, seed=0)
    assert len(tune) == 50 and len(evl) == 50
    assert set(tune).isdisjoint(set(evl))
    assert set(tune) | set(evl) == set(words)


def test_split_is_seed_deterministic():
    words = [f"w{i:04d}" for i in range(100)]
    a, _ = split_secrets(words, seed=1)
    b, _ = split_secrets(words, seed=1)
    assert a == b
