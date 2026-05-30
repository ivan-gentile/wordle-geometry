"""Load GloVe vectors for a given vocabulary (semantic control).

Streams a GloVe .txt file, keeping only the words we need. Returns a dict
word -> vector. Words absent from GloVe are simply omitted; the semantic
geometry builder fills them with a seeded random fallback.
"""
import os
from typing import Dict, List

import numpy as np


def load_glove_for_vocab(glove_txt: str, vocab: List[str]) -> Dict[str, np.ndarray]:
    want = set(vocab)
    out: Dict[str, np.ndarray] = {}
    with open(glove_txt, "r", encoding="utf-8") as fh:
        for line in fh:
            sp = line.rstrip().split(" ")
            w = sp[0]
            if w in want:
                out[w] = np.asarray(sp[1:], dtype=np.float64)
                if len(out) == len(want):
                    break
    return out
