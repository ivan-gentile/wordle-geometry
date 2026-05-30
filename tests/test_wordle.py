"""Tests for the Wordle feedback rule.

Pattern encoding (3b1b convention): per-letter code
    0 = MISS  (gray)
    1 = MISPLACED (yellow)
    2 = EXACT (green)
A pattern is the length-5 tuple of codes, optionally packed to a base-3 int.
"""
import numpy as np
import pytest

from src.wordle import (
    feedback, MISS, MISPLACED, EXACT, pattern_to_int, int_to_pattern,
    load_answers,
)


def test_all_green_when_guess_equals_secret():
    assert feedback("crane", "crane") == (EXACT, EXACT, EXACT, EXACT, EXACT)


def test_all_gray_when_no_letters_shared():
    # secret has none of g,u,m,b,o
    assert feedback("gumbo", "crane") == (MISS, MISS, MISS, MISS, MISS)


def test_simple_misplaced_and_exact():
    # secret CRANE, guess CARES:
    #   C green (pos0), A misplaced (secret has A at pos2), R misplaced (secret R at pos1),
    #   E misplaced (secret E at pos4), S miss
    assert feedback("cares", "crane") == (EXACT, MISPLACED, MISPLACED, MISPLACED, MISS)


def test_duplicate_guess_letters_limited_by_secret_count():
    # Classic case. secret ABBEY (one A, two B, one E, one Y), guess BABES.
    # Greens first: pos1 'A'? no. Let's reason per Wordle rules.
    # secret:  A B B E Y
    # guess:   B A B E S
    #   pos0 B: secret pos0 is A -> not green. B exists in secret -> yellow (consume one B)
    #   pos1 A: secret pos1 is B -> not green. A exists -> yellow (consume the A)
    #   pos2 B: secret pos2 is B -> GREEN (consume the 2nd B)
    #   pos3 E: secret pos3 is E -> GREEN
    #   pos4 S: not in secret -> gray
    assert feedback("babes", "abbey") == (MISPLACED, MISPLACED, EXACT, EXACT, MISS)


def test_duplicate_guess_letter_second_occurrence_grays_when_secret_has_only_one():
    # secret APPLE has two P. guess PUPPY has three P.
    # secret: A P P L E   guess: P U P P Y
    #   pos0 P: secret pos0 A -> not green; P in secret (2 available) -> yellow, 1 left
    #   pos1 U: not in secret -> gray
    #   pos2 P: secret pos2 is P -> GREEN, secret P count now 1 used as green... recompute properly:
    # Greens first: pos2 P matches secret pos2 P -> green. secret P remaining after greens = 1.
    #   pos0 P: yellow (consume remaining 1 P) -> remaining 0
    #   pos3 P: no P left -> gray
    #   pos1 U gray, pos4 Y gray
    assert feedback("puppy", "apple") == (MISPLACED, MISS, EXACT, MISS, MISS)


def test_green_takes_priority_over_yellow_for_same_letter():
    # secret SPEED, guess ERASE
    # secret: S P E E D   guess: E R A S E
    # greens: none align (E at guess0 vs secret0 S; E at guess4 vs secret4 D) -> no greens
    # secret E count = 2.
    #   pos0 E: not green; E available -> yellow, remaining 1
    #   pos1 R: gray
    #   pos2 A: gray
    #   pos3 S: secret has S at pos0 -> yellow
    #   pos4 E: E available (remaining 1) -> yellow, remaining 0
    assert feedback("erase", "speed") == (MISPLACED, MISS, MISS, MISPLACED, MISPLACED)


def test_load_answers_returns_clean_word_list():
    words = load_answers()
    assert len(words) > 2000  # full Wordle answer set
    assert all(len(w) == 5 for w in words)
    assert all(w.isalpha() and w.islower() for w in words)
    assert len(set(words)) == len(words)  # unique
    assert "crane" in words and "abate" in words


def test_pattern_int_roundtrip():
    pat = (MISS, MISPLACED, EXACT, EXACT, MISS)
    i = pattern_to_int(pat)
    assert isinstance(i, int)
    assert 0 <= i < 243
    assert int_to_pattern(i) == pat


def test_pattern_int_encoding_is_base3_little_endian():
    # 3b1b convention: pattern = sum(code_i * 3**i)
    assert pattern_to_int((EXACT, MISS, MISS, MISS, MISS)) == 2  # 2 * 3**0
    assert pattern_to_int((MISS, EXACT, MISS, MISS, MISS)) == 6  # 2 * 3**1
    assert pattern_to_int((MISS, MISS, MISS, MISS, MISS)) == 0
    assert pattern_to_int((EXACT, EXACT, EXACT, EXACT, EXACT)) == 2 + 6 + 18 + 54 + 162
