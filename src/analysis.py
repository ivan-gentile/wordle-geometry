"""M3 analysis diagnostics.

mean_convergence_curve
    Average per-turn ||p - E[secret]|| across games. A monotone-decreasing
    curve is the mechanistic evidence that "moving downhill" is real -- the
    position genuinely converges toward the secret embedding.

cell_convexity_score
    For a fixed probe guess g, the secrets partition into consistency cells
    {a : P[g,a]=f}. The silhouette of this clustering in E measures how
    geometrically separated/convex the cells are -- ties the WHY (good geometry)
    to the WHAT (winning play). Higher = cells are tighter and better separated.
"""
from typing import List

import numpy as np


def mean_convergence_curve(traces: List[List[float]], max_len: int = 6) -> List[float]:
    """Average distance-to-secret at each turn across games (ragged-safe).

    Turn t averages over only the games that reached turn t.
    """
    curve = []
    for t in range(max_len):
        vals = [tr[t] for tr in traces if len(tr) > t]
        curve.append(float(np.mean(vals)) if vals else float("nan"))
    return curve


def cell_convexity_score(P: np.ndarray, E: np.ndarray, probe_idx: int) -> float:
    """Silhouette of the probe's consistency-cell clustering in E.

    Labels each secret by the pattern P[probe, secret]; computes the silhouette
    score of these labels in the embedding E. Cells with <2 members or a single
    label are handled by sklearn's requirements (we drop singletons' effect by
    requiring >=2 distinct labels with >=2 members each).
    """
    from sklearn.metrics import silhouette_score

    labels = P[probe_idx].astype(int)
    # silhouette needs >=2 clusters and every sample in a cluster of size>=1;
    # restrict to labels that occur at least twice for a meaningful score.
    counts = np.bincount(labels)
    keep_labels = np.where(counts >= 2)[0]
    mask = np.isin(labels, keep_labels)
    if len(np.unique(labels[mask])) < 2 or mask.sum() < 3:
        return float("nan")
    return float(silhouette_score(E[mask], labels[mask]))
