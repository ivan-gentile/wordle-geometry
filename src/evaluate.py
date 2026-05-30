"""Evaluation harness: run agents over a held-out secret set, collect metrics.

Honesty: hyperparameters (eta, d) are swept on the TUNE split and the headline
numbers are reported on the disjoint EVAL split. The geometry is built from S
over the full vocabulary (a fixed structural fact, like the alphabet -- not
test leakage); only tunable knobs are split-protected.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.agents import (
    play_geometry_game, random_agent_game, freq_greedy_game, GameResult,
)
from src.controller import WordleNavigator


@dataclass
class Metrics:
    name: str
    n: int
    win_rate: float            # within the guess budget
    mean_guesses_won: float    # mean over won games
    mean_guesses_all: float    # losses counted as (budget + 1)
    hist: Dict[int, int]       # guess-count histogram (won games)


def summarize(name: str, results: List[GameResult], max_guesses: int) -> Metrics:
    won = [r for r in results if r.won]
    win_rate = len(won) / len(results)
    mg_won = float(np.mean([r.n_guesses for r in won])) if won else float("nan")
    mg_all = float(np.mean([
        r.n_guesses if r.won else (max_guesses + 1) for r in results
    ]))
    hist: Dict[int, int] = {}
    for r in won:
        hist[r.n_guesses] = hist.get(r.n_guesses, 0) + 1
    return Metrics(name, len(results), win_rate, mg_won, mg_all, hist)


def split_secrets(words: List[str], frac_tune: float = 0.5, seed: int = 0
                  ) -> Tuple[List[str], List[str]]:
    """Disjoint tune/eval split of the answer words."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(words))
    cut = int(len(words) * frac_tune)
    tune = [words[i] for i in perm[:cut]]
    evl = [words[i] for i in perm[cut:]]
    return tune, evl


def eval_geometry_agent(
    nav: WordleNavigator, words: List[str], secrets: List[str],
    eta: float, mode: str = "cell", max_guesses: int = 6, name: str = "geometry",
) -> Tuple[Metrics, List[GameResult]]:
    results = [
        play_geometry_game(nav, words, s, max_guesses=max_guesses, eta=eta, mode=mode)
        for s in secrets
    ]
    return summarize(name, results, max_guesses), results


def eval_random_agent(words, secrets, max_guesses=6, seed=0) -> Metrics:
    results = [random_agent_game(words, s, max_guesses, seed=seed + i)
               for i, s in enumerate(secrets)]
    return summarize("random-guess", results, max_guesses)


def eval_freq_greedy(words, secrets, max_guesses=6) -> Metrics:
    results = [freq_greedy_game(words, s, max_guesses) for s in secrets]
    return summarize("freq-greedy", results, max_guesses)
