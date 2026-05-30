"""Relation-B similarity between answer words.

S[a, b] = fraction of guesses g for which secrets a and b receive the SAME
feedback pattern = mean_g 1[P[g, a] == P[g, b]].

S ~ 1 => words are nearly indistinguishable by feedback (confusable answers);
S ~ 0 => almost every guess separates them. This is the "which beliefs is the
answer consistent with" geometry that the position vector moves through.
"""
import numpy as np


def relation_b_similarity(P: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Compute the (S, S) Relation-B similarity from pattern matrix P (G, S).

    Exact and vectorized: process guesses in blocks, broadcast-compare each
    block's pattern columns pairwise, and accumulate the agreement count.
    ``S[a, b] = (1/G) sum_g 1[P[g, a] == P[g, b]]``.

    For the full 2,315-word set this runs in ~10s with a ~340 MB transient per
    block (tunable via ``chunk``).
    """
    G, S = P.shape
    agree = np.zeros((S, S), dtype=np.int32)
    for start in range(0, G, chunk):
        block = P[start:start + chunk]                       # (g, S) uint8
        eq = block[:, :, None] == block[:, None, :]          # (g, S, S) bool
        agree += eq.sum(axis=0, dtype=np.int32)
    return (agree / G).astype(np.float64)
