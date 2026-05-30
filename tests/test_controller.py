"""Tests for the frozen controller: 243-arrow table + strict play-time rule.

The honesty contract (spec §3): at play-time the controller sees only
(position, guess_index, pattern). It never touches the candidate set, never
recomputes consistency, never reads P. Enforced structurally by a test that the
play module imports neither src.patterns nor src.similarity.
"""
import ast
import os

import numpy as np
import pytest

from src.wordle import N_PATTERNS, pattern_to_int, feedback, EXACT
from src.patterns import generate_pattern_matrix
from src.geometries import random_geometry, mds_geometry
from src.similarity import relation_b_similarity
from src.controller import (
    build_arrow_table, WordleNavigator, build_cell_centroid_table,
)


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade"]


def _setup(geometry="mds", d=8):
    P = generate_pattern_matrix(WORDS)
    if geometry == "mds":
        E = mds_geometry(relation_b_similarity(P), d=d, seed=0)
    else:
        E = random_geometry(len(WORDS), d=d, seed=0)
    table = build_arrow_table(P, E)
    return P, E, table


def _nav_arrow(P, E, table):
    return WordleNavigator(E, arrow_table=table)


def test_arrow_table_has_one_entry_per_pattern():
    P, E, table = _setup()
    assert table.shape == (N_PATTERNS, E.shape[1])
    assert np.isfinite(table).all()


def test_navigator_initial_position_is_centroid():
    P, E, table = _setup()
    nav = WordleNavigator(E, arrow_table=table)
    assert np.allclose(nav.initial_position(), E.mean(axis=0))


def test_nearest_word_returns_index_of_closest_embedding():
    P, E, table = _setup()
    nav = WordleNavigator(E, arrow_table=table)
    # a position exactly at word k must read back word k
    for k in [0, 3, 7]:
        assert nav.nearest_word(E[k]) == k


def test_move_changes_position_using_only_pattern_and_guess():
    P, E, table = _setup()
    nav = WordleNavigator(E, arrow_table=table)
    p0 = nav.initial_position()
    pat = pattern_to_int(feedback("slate", "crane"))
    p1 = nav.move(p0, guess_idx=WORDS.index("slate"), pattern=pat, eta=0.5)
    assert p1.shape == p0.shape
    assert not np.allclose(p1, p0)  # it actually moved


def test_all_green_pattern_pulls_toward_the_guess_itself():
    # If feedback is all-green for guess g, the secret IS g, and the
    # guess-relative arrow delta_f for the all-green pattern should be ~0, so the
    # update pulls the position toward E[g] (since target = E[g] + ~0).
    P, E, table = _setup()
    nav = WordleNavigator(E, arrow_table=table)
    g = WORDS.index("crane")
    all_green = pattern_to_int((EXACT,) * 5)
    p0 = nav.initial_position()
    p1 = nav.move(p0, guess_idx=g, pattern=all_green, eta=1.0)
    # with eta=1.0 the position jumps to E[g] + delta_allgreen ~= E[g]
    assert np.linalg.norm(p1 - E[g]) < np.linalg.norm(p0 - E[g])


def test_cell_centroid_table_shape():
    P, E, _ = _setup()
    T = build_cell_centroid_table(P, E)
    assert T.shape == (len(WORDS), N_PATTERNS, E.shape[1])


def test_cell_centroid_table_matches_true_cell_centroid():
    # T[g, f] must equal the centroid of {a : P[g, a] == f} in E.
    P, E, _ = _setup()
    T = build_cell_centroid_table(P, E)
    g = WORDS.index("slate")
    f = pattern_to_int(feedback("slate", "crane"))
    cell = np.where(P[g] == f)[0]
    expected = E[cell].mean(axis=0)
    assert np.allclose(T[g, f], expected)


def test_all_green_cell_centroid_is_the_guess_itself():
    # The only secret consistent with an all-green from guess g is g itself,
    # so T[g, all_green] == E[g].
    P, E, _ = _setup()
    T = build_cell_centroid_table(P, E)
    g = WORDS.index("crane")
    all_green = pattern_to_int((EXACT,) * 5)
    assert np.allclose(T[g, all_green], E[g])


def test_move_to_cell_pulls_toward_secret_for_consistent_guess():
    # With the cell-centroid target, moving after a guess must reduce the
    # distance to the secret embedding (the bug that the pattern-only arrow had).
    P, E, _ = _setup()
    T = build_cell_centroid_table(P, E)
    nav = WordleNavigator(E, arrow_table=None, cell_table=T)
    secret = "crane"
    si = WORDS.index(secret)
    g = WORDS.index("slate")
    f = pattern_to_int(feedback("slate", secret))
    p0 = nav.initial_position()
    p1 = nav.move_to_cell(p0, guess_idx=g, pattern=f, eta=1.0)
    assert np.linalg.norm(p1 - E[si]) <= np.linalg.norm(p0 - E[si])


def test_play_module_does_not_import_consistency_code():
    # Structural enforcement of the honesty contract: the controller's play-time
    # path must not import src.patterns or src.similarity.
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "controller.py")
    tree = ast.parse(open(path).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.name)
    assert "src.patterns" not in imported, "controller must not import P at play-time"
    assert "src.similarity" not in imported, "controller must not import S at play-time"
