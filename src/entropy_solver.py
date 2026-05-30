"""Constrained entropy solver: the 3b1b method, answers-only, greedy one-step.

This is the UPPER REFERENCE -- the "expensive search every move" agent. Unlike
the geometry agent, it is explicitly allowed to read P and filter the candidate
set every turn (that is the whole point: it quantifies how much competence the
geometry amortizes into a frozen lookup).

At each turn: over the current consistent candidate set, pick the guess that
maximizes the Shannon entropy of its induced pattern distribution; observe
feedback; restrict the candidate set to words consistent with that feedback.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.wordle import pattern_to_int, feedback, EXACT

ALL_GREEN = pattern_to_int((EXACT,) * 5)


@dataclass
class SolverResult:
    secret: str
    won: bool
    n_guesses: int
    guesses: List[str] = field(default_factory=list)


def _best_guess(P: np.ndarray, candidates: np.ndarray) -> int:
    """Index (into the full word list) of the max-entropy guess.

    Restrict guesses to the current candidate set (answers-only play). For each
    candidate guess g, the patterns it induces over the candidate secrets form a
    distribution; pick the g maximizing its entropy.
    """
    sub = P[np.ix_(candidates, candidates)]   # (C, C) patterns guess x secret
    best_h, best_g = -1.0, candidates[0]
    for row, g in enumerate(candidates):
        pats = sub[row]
        counts = np.bincount(pats, minlength=243).astype(np.float64)
        p = counts[counts > 0] / counts.sum()
        h = -(p * np.log2(p)).sum()
        if h > best_h:
            best_h, best_g = h, g
    return int(best_g)


def entropy_solver_game(
    P: np.ndarray, words: List[str], secret: str, max_guesses: int = 6
) -> SolverResult:
    candidates = np.arange(len(words))
    secret_idx = words.index(secret)
    guesses: List[str] = []

    for _ in range(max_guesses):
        if len(candidates) == 1:
            gi = int(candidates[0])
        else:
            gi = _best_guess(P, candidates)
        guesses.append(words[gi])

        pat = int(P[gi, secret_idx])
        if pat == ALL_GREEN:
            return SolverResult(secret, True, len(guesses), guesses)

        # restrict to words consistent with the observed feedback
        consistent = P[gi, candidates] == pat
        candidates = candidates[consistent]

    return SolverResult(secret, False, len(guesses), guesses)
