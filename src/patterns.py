"""Vectorized Wordle pattern matrix.

``generate_pattern_matrix(guesses, secrets)`` returns an array ``P`` with
``P[i, j]`` = the base-3-packed feedback pattern for guessing ``guesses[i]``
against secret ``secrets[j]``. Implements correct duplicate-letter handling
(greens first, then yellows consuming remaining secret-letter counts), fully
vectorized in numpy.
"""
import os
from typing import List, Optional

import numpy as np

from src.wordle import WORD_LEN, MISS, MISPLACED, EXACT


def _words_to_int_array(words: List[str]) -> np.ndarray:
    """(N, WORD_LEN) uint8 array of letter codes a..z -> 0..25."""
    arr = np.array([[ord(c) - ord("a") for c in w] for w in words], dtype=np.uint8)
    return arr


def generate_pattern_matrix(
    guesses: List[str], secrets: Optional[List[str]] = None
) -> np.ndarray:
    """Return packed feedback patterns for all (guess, secret) pairs.

    Shape ``(len(guesses), len(secrets))``, dtype uint8 (values 0..242).
    If ``secrets`` is None, uses ``guesses`` (square matrix).
    """
    if secrets is None:
        secrets = guesses

    g = _words_to_int_array(guesses)   # (G, 5)
    s = _words_to_int_array(secrets)   # (S, 5)
    G, S = g.shape[0], s.shape[0]

    # equality[i, j, p, q] = (guesses[i][p] == secrets[j][q])
    # Build the (G, S, 5, 5) letter-equality tensor.
    eq = g[:, None, :, None] == s[None, :, None, :]  # (G, S, 5, 5)

    codes = np.full((G, S, WORD_LEN), MISS, dtype=np.uint8)

    # Pass 1: greens (diagonal of the per-position equality).
    exact = np.einsum("gspp->gsp", eq.astype(np.uint8)).astype(bool)  # (G, S, 5)
    codes[exact] = EXACT

    # Remaining availability of each secret position: a secret letter is "used
    # up" if it was matched as a green. Track which secret positions are still
    # available to satisfy a yellow.
    secret_available = ~exact  # (G, S, 5) secret position q still free?
    # For greens, the guess at position p consumed secret position p.

    # Pass 2: yellows. For each guess position p (not green), find the first
    # available secret position q with the same letter; mark yellow and consume.
    # eq_pq[i,j,p,q] with greens removed from guess side handled by skipping
    # already-EXACT guess positions.
    eq_pq = eq.copy()
    # A secret position that is green (q where exact at q) cannot be reused.
    # secret_green[i,j,q] = exact[i,j,q]
    secret_green = exact  # green at position q means secret q is consumed
    eq_pq &= ~secret_green[:, :, None, :]  # zero out columns of consumed secrets

    for p in range(WORD_LEN):
        not_green = ~exact[:, :, p]                       # (G, S) guess pos p needs a color
        avail_match = eq_pq[:, :, p, :] & secret_available  # (G, S, 5) candidate secret cols
        has = avail_match.any(axis=-1) & not_green        # (G, S) can we place a yellow?
        # first available column index
        first = np.argmax(avail_match, axis=-1)           # (G, S) index of first True (0 if none)
        # apply yellow
        gi, sj = np.where(has)
        codes[gi, sj, p] = MISPLACED
        # consume that secret position
        secret_available[gi, sj, first[gi, sj]] = False

    # Pack base-3 little-endian.
    powers = (3 ** np.arange(WORD_LEN)).astype(np.uint16)  # (5,)
    packed = (codes.astype(np.uint16) * powers[None, None, :]).sum(axis=-1)
    return packed.astype(np.uint8)


def get_pattern_matrix(
    guesses: List[str],
    secrets: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Return the pattern matrix, loading from ``cache_path`` if present.

    The cache is keyed only by path; the caller is responsible for using a
    distinct path per (guesses, secrets) pair.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    P = generate_pattern_matrix(guesses, secrets)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, P)
    return P
