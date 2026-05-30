"""Tests for embedding-geometry builders.

A geometry is an (N, d) float array, row i = embedding of word i, aligned to
the answer word list. All builders share this contract so the controller and
agents are geometry-agnostic.
"""
import numpy as np
import pytest

from src.patterns import generate_pattern_matrix
from src.similarity import relation_b_similarity
from src.geometries import random_geometry, mds_geometry


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade"]


def _S():
    return relation_b_similarity(generate_pattern_matrix(WORDS))


def test_random_geometry_shape_and_seed_determinism():
    E1 = random_geometry(len(WORDS), d=8, seed=0)
    E2 = random_geometry(len(WORDS), d=8, seed=0)
    E3 = random_geometry(len(WORDS), d=8, seed=1)
    assert E1.shape == (len(WORDS), 8)
    assert np.array_equal(E1, E2)          # same seed -> identical
    assert not np.array_equal(E1, E3)      # different seed -> different


def test_mds_geometry_shape():
    E = mds_geometry(_S(), d=8, seed=0)
    assert E.shape == (len(WORDS), 8)
    assert np.isfinite(E).all()


def test_mds_preserves_similarity_order():
    # In the MDS embedding, a more B-similar pair should be (on average) closer
    # than a less B-similar pair. Check the rank correlation sign on a few pairs.
    S = _S()
    E = mds_geometry(S, d=8, seed=0)
    i_crane = WORDS.index("crane")
    i_trace = WORDS.index("trace")   # shares many letters with crane
    i_puppy = WORDS.index("puppy")   # shares little
    d_close = np.linalg.norm(E[i_crane] - E[i_trace])
    d_far = np.linalg.norm(E[i_crane] - E[i_puppy])
    # higher similarity => smaller distance
    assert (S[i_crane, i_trace] > S[i_crane, i_puppy]) == (d_close < d_far)


def test_mds_is_deterministic():
    S = _S()
    E1 = mds_geometry(S, d=8, seed=0)
    E2 = mds_geometry(S, d=8, seed=0)
    assert np.allclose(E1, E2)
