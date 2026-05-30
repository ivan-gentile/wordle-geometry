"""Agents that play Wordle, plus baselines.

The geometry agent is the experiment: it moves a position vector with the frozen
controller and reads back the nearest answer-word. It observes feedback exactly
as a human player does (via the game referee ``feedback``); it never filters the
candidate set by consistency.

Baselines:
    random_agent_game     -- guess uniformly at random (no repeats): the floor.
    freq_greedy_game      -- guess by static letter-frequency score: a weak
                             non-trivial floor.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.wordle import feedback, pattern_to_int, EXACT
from src.controller import WordleNavigator

ALL_GREEN = pattern_to_int((EXACT,) * 5)


@dataclass
class GameResult:
    secret: str
    won: bool
    n_guesses: int
    guesses: List[str] = field(default_factory=list)
    distance_trace: List[float] = field(default_factory=list)  # ||p - E[secret]|| per turn


def play_geometry_game(
    nav: WordleNavigator,
    words: List[str],
    secret: str,
    max_guesses: int = 6,
    eta: float = 0.7,
    mode: str = "cell",
) -> GameResult:
    """Play one game with the geometry-driven navigator.

    Loop: nearest word -> observe feedback -> move position. Stop on all-green
    or when the guess budget is exhausted. Records the per-turn distance from
    the position to the (unknown-to-the-agent) secret embedding for analysis.

    ``mode``:
        "cell"  -- main rule: move toward the frozen (guess, pattern) cell
                   centroid (build_cell_centroid_table).
        "arrow" -- sanity rung: move via the pattern-only displacement table
                   (build_arrow_table).
    """
    secret_idx = words.index(secret)
    p = nav.initial_position()
    tried = set()
    guesses: List[str] = []
    dist_trace: List[float] = []

    for _ in range(max_guesses):
        gi = nav.nearest_word(p, exclude=tried)
        tried.add(gi)
        guess = words[gi]
        guesses.append(guess)

        pat = pattern_to_int(feedback(guess, secret))
        if pat == ALL_GREEN:
            dist_trace.append(float(np.linalg.norm(p - nav.E[secret_idx])))
            return GameResult(secret, True, len(guesses), guesses, dist_trace)

        # move, then record distance of the NEW position to the secret embedding
        if mode == "cell":
            p = nav.move_to_cell(p, guess_idx=gi, pattern=pat, eta=eta)
        elif mode == "arrow":
            p = nav.move(p, guess_idx=gi, pattern=pat, eta=eta)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        dist_trace.append(float(np.linalg.norm(p - nav.E[secret_idx])))

    return GameResult(secret, False, len(guesses), guesses, dist_trace)


def random_agent_game(
    words: List[str], secret: str, max_guesses: int = 6, seed: int = 0
) -> GameResult:
    """Floor baseline: guess uniformly at random without repeats."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(words)))
    guesses: List[str] = []
    for k in range(min(max_guesses, len(words))):
        guess = words[order[k]]
        guesses.append(guess)
        if guess == secret:
            return GameResult(secret, True, len(guesses), guesses, [])
    return GameResult(secret, False, len(guesses), guesses, [])


def _letter_frequency_scores(words: List[str]) -> np.ndarray:
    """Score each word by summed positional letter frequency (distinct letters)."""
    from collections import Counter
    counts = Counter("".join(words))
    scores = np.array(
        [sum(counts[c] for c in set(w)) for w in words], dtype=np.float64
    )
    return scores


def freq_greedy_game(
    words: List[str], secret: str, max_guesses: int = 6
) -> GameResult:
    """Weak non-trivial baseline: always guess the highest letter-frequency word
    not yet tried. Static (no feedback use) -> a deliberately weak reference."""
    scores = _letter_frequency_scores(words)
    order = list(np.argsort(scores)[::-1])
    guesses: List[str] = []
    for k in range(min(max_guesses, len(words))):
        guess = words[order[k]]
        guesses.append(guess)
        if guess == secret:
            return GameResult(secret, True, len(guesses), guesses, [])
    return GameResult(secret, False, len(guesses), guesses, [])
