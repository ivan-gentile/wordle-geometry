"""Tests for the contrastive geometry trainers (M2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.patterns import generate_pattern_matrix
from src.similarity import relation_b_similarity
from src.contrastive import train_reconstruction, train_supcon, reconstruction_loss_value


WORDS = ["crane", "crate", "abbey", "babes", "apple", "puppy", "speed", "erase",
         "slate", "trace", "grace", "brace", "place", "plate", "plane", "blade",
         "shade", "spade", "scale", "stale", "stare", "scare", "snare", "share"]


def _S():
    return relation_b_similarity(generate_pattern_matrix(WORDS))


def test_reconstruction_output_shape_and_finite():
    E = train_reconstruction(_S(), d=8, epochs=50, seed=0)
    assert E.shape == (len(WORDS), 8)
    assert np.isfinite(E).all()


def test_reconstruction_reduces_fit_loss():
    # Training must lower the S-reconstruction loss vs the random init.
    S = _S()
    rng = np.random.default_rng(0)
    E_init = rng.standard_normal((len(WORDS), 8)).astype(np.float64)
    loss_init = reconstruction_loss_value(E_init, S)
    E_trained = train_reconstruction(S, d=8, epochs=300, seed=0)
    loss_trained = reconstruction_loss_value(E_trained, S)
    assert loss_trained < loss_init


def test_reconstruction_is_seed_deterministic():
    S = _S()
    E1 = train_reconstruction(S, d=8, epochs=50, seed=0)
    E2 = train_reconstruction(S, d=8, epochs=50, seed=0)
    assert np.allclose(E1, E2)


def test_supcon_output_shape_and_finite():
    E = train_supcon(_S(), d=8, epochs=50, seed=0, k_pos=3, n_neg=8)
    assert E.shape == (len(WORDS), 8)
    assert np.isfinite(E).all()


def test_supcon_brings_high_s_neighbors_closer_than_low_s():
    # After SupCon training, a high-S pair should be closer (cosine) than a
    # low-S pair on average. Check the sign on a clear case.
    S = _S()
    E = train_supcon(S, d=8, epochs=400, seed=0, k_pos=3, n_neg=8)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    cos = En @ En.T
    n = len(WORDS)
    iu = np.triu_indices(n, k=1)
    s_flat = S[iu]
    cos_flat = cos[iu]
    # Use clearly-separated, non-empty groups. On this small lexicon ~30% of
    # pairs have S=0, so a 0.25-quantile threshold can be empty; the top decile
    # vs the bottom half is robust and still a strong contrast.
    hi = s_flat >= np.quantile(s_flat, 0.90)
    lo = s_flat <= np.quantile(s_flat, 0.50)
    assert hi.any() and lo.any()
    assert cos_flat[hi].mean() > cos_flat[lo].mean()
