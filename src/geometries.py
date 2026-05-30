"""Embedding-geometry builders.

Each builder returns an (N, d) float array, row i = embedding of word i, aligned
to the answer word list. The controller and agents are geometry-agnostic: they
only ever see this table. Geometry-source is the experiment's independent
variable.

Builders:
    random_geometry  -- control: each word a random vector (ignores P/S)
    mds_geometry     -- baseline: classical MDS of D = 1 - S
    semantic_geometry-- control: external word vectors (GloVe), aligned to words
"""
from typing import Dict, List, Optional

import numpy as np


def random_geometry(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    """Control geometry: ``n`` i.i.d. standard-normal vectors in R^d."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


def mds_geometry(S: np.ndarray, d: int = 32, seed: int = 0) -> np.ndarray:
    """Classical (Torgerson) MDS embedding of the dissimilarity ``D = 1 - S``.

    Double-centers the squared-distance matrix and takes the top-``d``
    eigenvectors. Deterministic up to sign; ``seed`` is accepted for API
    symmetry and used only to fix sign conventions.
    """
    D = 1.0 - S
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n          # centering matrix
    B = -0.5 * J @ D2 @ J                         # double-centered Gram matrix
    B = (B + B.T) / 2                             # enforce symmetry (numerical)

    eigvals, eigvecs = np.linalg.eigh(B)          # ascending
    idx = np.argsort(eigvals)[::-1][:d]           # top-d
    vals = np.clip(eigvals[idx], a_min=0.0, a_max=None)
    vecs = eigvecs[:, idx]

    # Fix sign convention deterministically: make the largest-magnitude entry of
    # each component positive.
    for k in range(vecs.shape[1]):
        col = vecs[:, k]
        if col[np.argmax(np.abs(col))] < 0:
            vecs[:, k] = -col

    E = vecs * np.sqrt(vals)[None, :]
    return E.astype(np.float64)


def semantic_geometry(
    words: List[str], vectors: Dict[str, np.ndarray], d: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Control geometry from external word vectors (e.g. GloVe).

    ``vectors`` maps word -> raw vector. Words missing from ``vectors`` get a
    random fallback vector (seeded). If ``d`` is given, the result is reduced to
    ``d`` dims by PCA; otherwise the native dimensionality is kept.
    """
    rng = np.random.default_rng(seed)
    dim = len(next(iter(vectors.values())))
    rows = []
    n_missing = 0
    for w in words:
        v = vectors.get(w)
        if v is None:
            v = rng.standard_normal(dim)
            n_missing += 1
        rows.append(np.asarray(v, dtype=np.float64))
    E = np.vstack(rows)
    if n_missing:
        # caller may want to know; print is fine for a research script
        print(f"semantic_geometry: {n_missing}/{len(words)} words missing -> random fallback")

    if d is not None and d < E.shape[1]:
        Ec = E - E.mean(axis=0, keepdims=True)
        # PCA via SVD (deterministic)
        U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
        E = U[:, :d] * s[:d][None, :]
    return E.astype(np.float64)
