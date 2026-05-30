"""M4: information-aware readout with frozen per-word probe scores.

Diagnosis: the SupCon geometry agent is pointed correctly (wins 98% given 15
guesses) but too slow to fit inside 6. The myopic nearest-answer readout spends
guesses on low-information words near the position. Fix: bake a per-word probe
score from P ONCE at build time (entropy of the word's pattern distribution over
all answers), then at play-time read out the highest-scoring word among the
k-nearest to the position.

Honesty contract preserved: play-time uses only (position, frozen score vector).
The scores are a frozen per-word scalar table -- no P read, no candidate
filtering during play. ``build_probe_scores`` takes P as a build-time argument.
"""
from typing import Optional, Set

import numpy as np

from src.controller import N_PATTERNS, WordleNavigator


def build_probe_scores(P: np.ndarray) -> np.ndarray:
    """Per-word probe quality = Shannon entropy (bits) of the word's pattern
    distribution over all secrets. High = the word splits the answer space into
    many balanced patterns (an informative probe). Computed once at build time.
    """
    G, S = P.shape
    scores = np.zeros(G, dtype=np.float64)
    for g in range(G):
        counts = np.bincount(P[g], minlength=N_PATTERNS).astype(np.float64)
        p = counts[counts > 0] / counts.sum()
        scores[g] = -(p * np.log2(p)).sum()
    return scores


def informed_nearest(
    nav: WordleNavigator,
    p: np.ndarray,
    probe_scores: np.ndarray,
    k: int = 8,
    exclude: Optional[Set[int]] = None,
) -> int:
    """Among the k answers nearest ``p``, return the one with the highest frozen
    probe score (ties broken by proximity). With k=1 this is plain nearest-word.

    Uses only distances to E and the frozen ``probe_scores`` -- no P at play-time.
    """
    d2 = np.sum((nav.E - p[None, :]) ** 2, axis=1)
    if exclude:
        d2 = d2.copy()
        for i in exclude:
            d2[i] = np.inf
    # k nearest (finite) candidates
    k = max(1, min(k, np.isfinite(d2).sum()))
    cand = np.argpartition(d2, k - 1)[:k]
    cand = cand[np.isfinite(d2[cand])]
    if len(cand) == 0:
        return int(np.argmin(d2))
    # pick max probe score; tie-break by smaller distance
    best = cand[np.lexsort((d2[cand], -probe_scores[cand]))[0]]
    return int(best)
