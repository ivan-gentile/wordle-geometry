"""M4: information-aware readout among the k-nearest (frozen probe scores).

The bottleneck is guesses-per-game efficiency, not direction (SupCon wins 98% if
given 15 guesses). The fix: among the k answers nearest the position, prefer the
one with the highest BUILD-TIME probe-quality score (one scalar per word, baked
from P once). Play-time still reads only (position, frozen scores) -- no P.
"""
import numpy as np
import pytest

from src.wordle import load_answers
from src.patterns import generate_pattern_matrix
from src.controller import build_cell_centroid_table, WordleNavigator
from src.m4 import build_probe_scores


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade",
         "shade", "spade", "scale", "stale", "stare", "scare", "snare", "share"]


def test_probe_scores_shape_and_finite():
    P = generate_pattern_matrix(WORDS)
    scores = build_probe_scores(P)
    assert scores.shape == (len(WORDS),)
    assert np.isfinite(scores).all()


def test_probe_scores_rank_informative_words_higher():
    # A word that splits the answer set into many distinct patterns (high entropy)
    # should score higher than one that lumps answers together.
    P = generate_pattern_matrix(WORDS)
    scores = build_probe_scores(P)
    # 'slate'/'stare' share letters with many words -> informative; 'puppy' is
    # an outlier with repeated letters -> less informative on this list.
    assert scores[WORDS.index("slate")] > scores[WORDS.index("puppy")]


def test_informed_readout_falls_back_to_nearest_when_k_is_one():
    # With k=1 the informed readout must equal the plain nearest-word readout.
    P = generate_pattern_matrix(WORDS)
    E = np.random.default_rng(0).standard_normal((len(WORDS), 8))
    scores = build_probe_scores(P)
    nav = WordleNavigator(E, cell_table=build_cell_centroid_table(P, E))
    from src.m4 import informed_nearest
    p = E.mean(0)
    assert informed_nearest(nav, p, scores, k=1) == nav.nearest_word(p)


def test_informed_readout_does_not_pick_excluded_words():
    P = generate_pattern_matrix(WORDS)
    E = np.random.default_rng(1).standard_normal((len(WORDS), 8))
    scores = build_probe_scores(P)
    nav = WordleNavigator(E, cell_table=build_cell_centroid_table(P, E))
    from src.m4 import informed_nearest
    p = E.mean(0)
    tried = {0, 1, 2, 3, 4}
    pick = informed_nearest(nav, p, scores, k=5, exclude=tried)
    assert pick not in tried


def test_play_game_accepts_a_readout_hook():
    # play_geometry_game should accept a custom readout(nav, p, exclude) callable;
    # passing the informed readout must still produce a valid, budget-respecting game.
    from functools import partial
    from src.geometries import mds_geometry
    from src.similarity import relation_b_similarity
    from src.agents import play_geometry_game
    from src.m4 import informed_nearest

    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    E = mds_geometry(S, d=8, seed=0)
    scores = build_probe_scores(P)
    nav = WordleNavigator(E, cell_table=build_cell_centroid_table(P, E))
    readout = lambda nav, p, exclude: informed_nearest(nav, p, scores, k=5, exclude=exclude)
    res = play_geometry_game(nav, WORDS, "crane", max_guesses=6, eta=0.8,
                             mode="cell", readout=readout)
    assert 1 <= res.n_guesses <= 6
    assert len(set(res.guesses)) == len(res.guesses)
