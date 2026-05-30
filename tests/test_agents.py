"""Tests for agents: the geometry-driven play loop and baselines."""
import numpy as np
import pytest

from src.wordle import feedback, pattern_to_int, EXACT
from src.patterns import generate_pattern_matrix
from src.geometries import random_geometry, mds_geometry
from src.similarity import relation_b_similarity
from src.controller import build_arrow_table, build_cell_centroid_table, WordleNavigator
from src.agents import play_geometry_game, GameResult, random_agent_game


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade"]


def _nav(geometry="mds", d=8):
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    E = mds_geometry(S, d=d, seed=0) if geometry == "mds" else random_geometry(len(WORDS), d, 0)
    return WordleNavigator(
        E,
        arrow_table=build_arrow_table(P, E),
        cell_table=build_cell_centroid_table(P, E),
    )


def test_cell_mode_solves_a_small_game():
    # On the small lexicon, the cell-centroid mode (main rule) should win the
    # game for a secret in the list within the budget.
    nav = _nav()
    res = play_geometry_game(nav, WORDS, secret="crane", max_guesses=6, eta=0.8, mode="cell")
    assert res.won
    assert res.guesses[-1] == "crane"


def test_game_result_fields():
    nav = _nav()
    res = play_geometry_game(nav, WORDS, secret="crane", max_guesses=6, eta=0.7)
    assert isinstance(res, GameResult)
    assert isinstance(res.won, bool)
    assert 1 <= res.n_guesses <= 6
    assert len(res.guesses) == res.n_guesses
    assert len(res.distance_trace) == res.n_guesses  # ||p - E[secret]|| per turn


def test_win_is_detected_when_guess_equals_secret():
    nav = _nav()
    # Force a win: if the agent ever guesses the secret, the all-green pattern
    # must register a win and stop.
    res = play_geometry_game(nav, WORDS, secret="crane", max_guesses=6, eta=0.7)
    if res.won:
        assert res.guesses[-1] == "crane"
        last_pat = pattern_to_int(feedback(res.guesses[-1], "crane"))
        assert last_pat == pattern_to_int((EXACT,) * 5)


def test_does_not_repeat_guesses():
    nav = _nav()
    res = play_geometry_game(nav, WORDS, secret="puppy", max_guesses=6, eta=0.7)
    assert len(set(res.guesses)) == len(res.guesses)  # no repeats


def test_max_guesses_respected():
    nav = _nav(geometry="random")  # random geometry may fail to converge
    res = play_geometry_game(nav, WORDS, secret="abbey", max_guesses=6, eta=0.7)
    assert res.n_guesses <= 6


def test_random_agent_eventually_wins_with_enough_guesses():
    # Sanity for the floor baseline: with max_guesses = full list, it must win.
    res = random_agent_game(WORDS, secret="speed", max_guesses=len(WORDS), seed=0)
    assert res.won
    assert res.guesses[-1] == "speed"
