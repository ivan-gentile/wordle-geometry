"""Frozen controller: build-time lookup tables + strict play-time movement.

Honesty contract (spec §3):
  * The play-time object (WordleNavigator) sees only (position, guess_index,
    pattern). It never touches the surviving-candidate set, never recomputes
    consistency, never reads the pattern matrix P at play-time.
  * Movement is a frozen lookup baked ONCE at build time from (P, E). Zero
    learned play-time parameters.

This module must NOT import src.patterns or src.similarity (enforced by test).
Build-time functions receive P as an argument; they do not import it.

Two frozen targets are provided:
  * build_arrow_table(P, E)  -> (243, d): pattern-only mean displacement
      delta_f = mean_{(g,a): P[g,a]=f}(E[a]-E[g]). This is the context-free
      "sanity rung" (spec §4d) -- it averages over all guesses and is lossy.
  * build_cell_centroid_table(P, E) -> (N, 243, d): the (guess, pattern) cell
      centroid T[g,f] = centroid(E[{a : P[g,a]=f}]). This is the main target:
      it points at where THIS guess's feedback says the answer lies. Still a
      frozen build-time table keyed by (g, f); play-time reads only (p, g, f).
"""
from typing import Optional

import numpy as np

N_PATTERNS = 243  # inlined constant; avoids importing game internals here


def build_arrow_table(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Context-free pattern->direction table (sanity rung).

    delta_f = mean_{(g,a): P[g,a]=f}(E[a]-E[g]). Empty patterns -> zero arrow.
    """
    G, S = P.shape
    d = E.shape[1]
    sums = np.zeros((N_PATTERNS, d), dtype=np.float64)
    counts = np.zeros(N_PATTERNS, dtype=np.int64)
    for g in range(G):
        disp = E - E[g][None, :]
        np.add.at(sums, P[g], disp)
        np.add.at(counts, P[g], 1)
    nz = counts > 0
    table = np.zeros((N_PATTERNS, d), dtype=np.float64)
    table[nz] = sums[nz] / counts[nz][:, None]
    return table


def build_cell_centroid_table(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Main frozen target: (guess, pattern) consistency-cell centroid.

    ``T[g, f]`` = mean of ``E[a]`` over all secrets ``a`` with ``P[g, a] == f``
    (the consistency cell of guess ``g`` under feedback ``f``). Empty cells are
    zero. Shape ``(N, 243, d)``, computed once at build time.

    Interpretation: after guessing ``g`` and seeing ``f``, the answer lies in
    the cell ``{a : P[g,a]=f}``; its centroid is the best fixed estimate of
    where to move. Keyed by (g, f) only, so it is a legal frozen lookup.
    """
    G, S = P.shape
    d = E.shape[1]
    T = np.zeros((G, N_PATTERNS, d), dtype=np.float64)
    for g in range(G):
        sums = np.zeros((N_PATTERNS, d), dtype=np.float64)
        counts = np.zeros(N_PATTERNS, dtype=np.int64)
        np.add.at(sums, P[g], E)
        np.add.at(counts, P[g], 1)
        nz = counts > 0
        T[g, nz] = sums[nz] / counts[nz][:, None]
    return T


class WordleNavigator:
    """Play-time agent: a position vector moved by frozen tables only.

    Holds the frozen geometry ``E`` (the inert word->vector dictionary) and one
    or both frozen targets. The only play-time inputs are the current position,
    the index of the word just guessed, and the observed pattern.
    """

    def __init__(
        self,
        E: np.ndarray,
        arrow_table: Optional[np.ndarray] = None,
        cell_table: Optional[np.ndarray] = None,
    ):
        self.E = E
        self.arrow_table = arrow_table
        self.cell_table = cell_table

    def initial_position(self) -> np.ndarray:
        """Belief starts at the centroid of the geometry (maximally uncommitted)."""
        return self.E.mean(axis=0)

    def nearest_word(self, p: np.ndarray, exclude: Optional[set] = None) -> int:
        """Index of the answer-word whose embedding is closest to ``p``.

        ``exclude`` forbids re-guessing tried words (play-time bookkeeping, not a
        consistency filter -- it never consults patterns).
        """
        d2 = np.sum((self.E - p[None, :]) ** 2, axis=1)
        if exclude:
            d2 = d2.copy()
            for i in exclude:
                d2[i] = np.inf
        return int(np.argmin(d2))

    def move(self, p: np.ndarray, guess_idx: int, pattern: int, eta: float) -> np.ndarray:
        """Context-free (sanity-rung) update using the pattern-only arrow table.

            p <- (1 - eta) * p + eta * (E[guess] + delta_pattern)
        """
        if self.arrow_table is None:
            raise ValueError("arrow_table not provided")
        target = self.E[guess_idx] + self.arrow_table[pattern]
        return (1.0 - eta) * p + eta * target

    def move_to_cell(
        self, p: np.ndarray, guess_idx: int, pattern: int, eta: float
    ) -> np.ndarray:
        """Main update: move toward the frozen (guess, pattern) cell centroid.

            p <- (1 - eta) * p + eta * T[guess, pattern]

        Reads only the frozen cell table at (guess_idx, pattern); no P, no search.
        """
        if self.cell_table is None:
            raise ValueError("cell_table not provided")
        target = self.cell_table[guess_idx, pattern]
        return (1.0 - eta) * p + eta * target
