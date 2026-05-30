"""Wordle feedback rule and pattern (de)serialization.

Pattern codes follow the 3b1b convention:
    MISS=0 (gray), MISPLACED=1 (yellow), EXACT=2 (green).
A pattern is a length-5 tuple of codes. It packs to a base-3 integer in
[0, 243) via ``pattern = sum(code_i * 3**i)`` (little-endian).

The feedback rule implements the real Wordle duplicate-letter semantics:
greens are assigned first; remaining secret letter counts are then consumed
left-to-right by yellows.
"""
import os
from collections import Counter
from typing import List, Tuple

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ANSWERS_FILE = os.path.join(_DATA_DIR, "answers.txt")

MISS = 0
MISPLACED = 1
EXACT = 2

WORD_LEN = 5
N_PATTERNS = 3 ** WORD_LEN  # 243

Pattern = Tuple[int, int, int, int, int]


def load_answers(path: str = _ANSWERS_FILE) -> List[str]:
    """Load the canonical Wordle answer list (2,315 original answers).

    This is the full original answer set used by 3b1b; some later references
    cite 2,309 after the NYT removed a handful of words.
    """
    with open(path) as fh:
        return [line.strip().lower() for line in fh if line.strip()]


def feedback(guess: str, secret: str) -> Pattern:
    """Return the Wordle color pattern for ``guess`` against ``secret``.

    Two-pass algorithm with correct duplicate handling:
    1. Mark exact (green) matches and decrement available secret-letter counts.
    2. Left-to-right, mark misplaced (yellow) for letters still available.
    """
    guess = guess.lower()
    secret = secret.lower()
    if len(guess) != WORD_LEN or len(secret) != WORD_LEN:
        raise ValueError("words must be length 5")

    codes = [MISS] * WORD_LEN
    remaining = Counter(secret)

    # Pass 1: greens.
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            codes[i] = EXACT
            remaining[g] -= 1

    # Pass 2: yellows, consuming remaining counts left-to-right.
    for i, g in enumerate(guess):
        if codes[i] == EXACT:
            continue
        if remaining[g] > 0:
            codes[i] = MISPLACED
            remaining[g] -= 1

    return tuple(codes)


def pattern_to_int(pattern: Pattern) -> int:
    """Pack a length-5 code tuple to a base-3 integer (little-endian)."""
    value = 0
    for i, code in enumerate(pattern):
        value += code * (3 ** i)
    return value


def int_to_pattern(value: int) -> Pattern:
    """Unpack a base-3 integer back to a length-5 code tuple."""
    codes = []
    for _ in range(WORD_LEN):
        codes.append(value % 3)
        value //= 3
    return tuple(codes)
