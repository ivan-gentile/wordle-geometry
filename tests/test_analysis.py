"""Tests for the M3 analysis diagnostics."""
import numpy as np
import pytest

from src.patterns import generate_pattern_matrix
from src.similarity import relation_b_similarity
from src.geometries import mds_geometry, random_geometry
from src.analysis import mean_convergence_curve, cell_convexity_score


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade",
         "shade", "spade", "scale", "stale", "stare", "scare", "snare", "share"]


def test_convergence_curve_shape():
    traces = [[0.7, 0.5, 0.3], [0.8, 0.6], [0.9, 0.7, 0.5, 0.2]]
    curve = mean_convergence_curve(traces, max_len=4)
    assert len(curve) == 4
    # turn 1 mean of [0.7,0.8,0.9]
    assert abs(curve[0] - 0.8) < 1e-9


def test_convergence_curve_ignores_missing_turns():
    traces = [[0.5], [0.4, 0.2]]
    curve = mean_convergence_curve(traces, max_len=2)
    assert abs(curve[0] - 0.45) < 1e-9   # mean[0.5,0.4]
    assert abs(curve[1] - 0.2) < 1e-9    # only second trace has turn 2


def test_cell_convexity_higher_for_structured_geometry():
    # The feedback-aware MDS geometry should have more separated pattern cells
    # (higher silhouette) than a random geometry, for a fixed probe guess.
    P = generate_pattern_matrix(WORDS)
    S = relation_b_similarity(P)
    E_mds = mds_geometry(S, d=8, seed=0)
    E_rnd = random_geometry(len(WORDS), d=8, seed=0)
    probe = WORDS.index("slate")
    sil_mds = cell_convexity_score(P, E_mds, probe)
    sil_rnd = cell_convexity_score(P, E_rnd, probe)
    assert sil_mds > sil_rnd
